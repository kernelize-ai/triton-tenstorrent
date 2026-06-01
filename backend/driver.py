import functools
import triton
import os
import subprocess
import tempfile
import time
import platform
import importlib
from pathlib import Path

from triton.runtime.build import compile_module_from_src
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

try:
    from triton.runtime.jit import TensorDescriptor
except ImportError:
    from triton.tools.tensor_descriptor import TensorDescriptor


@functools.lru_cache()
def is_macos():
    return platform.system() == "Darwin"


@functools.lru_cache()
def external_openmp_path():
    return os.environ.get("TRITON_LOCAL_LIBOMP_PATH", "/opt/homebrew/opt/libomp/")


@functools.lru_cache()
def external_boost_path():
    return os.environ.get("TRITON_LOCAL_BOOST_PATH", "/opt/homebrew")


dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")] + [
    os.path.join(external_openmp_path(), "include") if is_macos() else []
] + [os.path.join(external_boost_path(), "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ["boost_fiber", "boost_context"]


@functools.lru_cache()
def system_ccflags():
    ccflags = ["-std=c++17"]
    if is_macos():
        ccflags.extend(["-undefined", "dynamic_lookup", "-Xclang"])
    ccflags.extend(["-fopenmp"])
    return ccflags


@functools.lru_cache()
def library_dirs():
    #lib_dirs = [_triton_C_dir]
    lib_dirs = []
    if is_macos():
        lib_dirs.extend([os.path.join(external_openmp_path(), "lib")])
        lib_dirs.extend([os.path.join(external_boost_path(), "lib")])
    return lib_dirs


def get_nexus_runtime():
    import nexus
    return nexus.get_runtime("tt-metal")


class CpuUtils(object):

    def __init__(self, runtime):
        self.runtime = runtime

    def load_binary(self, name, kernel, shared_mem, device):
        ## TODO: change to load_library from kernel string so the tmp files are not needed
        ## tmpfile must be persistent since the file will be jit compiled on the first run
        device = self.runtime.get_device(device)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".cpp", delete=False) as f:
            f.write(kernel)
            f.flush()
            os.fsync(f.fileno())
            os.stat(f.name)
            lib = device.load_library(f.name)
            kernel = lib.get_kernel(name)
            # TODO: properly handle num registers / max number threads
            return (lib, kernel, 1, shared_mem, 2**12)

    def get_device_properties(self, *args):
        # import nexus
        core_count = 130  # self.device.get_property_int(nexus.property.Size)
        return {
            "max_num_regs": core_count * 4, "max_shared_mem": 1024 * 1024 * 1024, "multiprocessor_count": core_count,
            "warpSize": 1
        }


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


class CPULauncher(object):

    def __init__(self, src, metadata):
        runtime = get_nexus_runtime()
        self.device = runtime.get_device(0)
        self.schedule = None
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        self.constants = {arg_idx(idx): value for idx, value in constants.items()}
        self.signature = {idx: value for idx, value in src.signature.items()}

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        import nexus
        #self.launch(gridX, gridY, gridZ, stream, function, *args)
        kernel_metadata = args[0]
        num_warps = kernel_metadata[0]
        # num_ctas = kernel_metadata[1] # should be 1
        shared_memory = kernel_metadata[2]
        # clusterDimX = kernel_metadata[3] # should be 1
        # clusterDimY = kernel_metadata[4] # should be 1
        # clusterDimZ = kernel_metadata[5] # should be 1
        launch_metadata = args[1]
        launch_enter_hook = args[2]
        launch_exit_hook = args[3]

        if self.schedule is None:
            self.schedule = self.device.create_schedule()
            schedule = self.schedule
            command = schedule.create_command(function)
            import torch
            ## TODO: Get CB depth from TuningConfig
            cb_depth = 1
            buffers = []
            sig_types = list(self.signature.values())
            idx = 0
            add_arg = lambda arg: (command.set_arg(idx, arg), idx + 1)
            cb_idx = 0
            for i, arg in enumerate(args[4:]):
                ty = sig_types[i]
                if ty == "constexpr":
                    continue
                if isinstance(arg, torch.Tensor) or isinstance(arg, nexus.buffer):
                    command.set_const(cb_idx, cb_depth, "CB", nexus.get_data_type(arg))
                    cb_idx += 1
                    arg = self.device.create_buffer(arg)
                    _, idx = add_arg(arg)
                    buffers.append(arg)
                elif isinstance(arg, TensorDescriptor):
                    arg_base = arg.base
                    command.set_const(cb_idx, cb_depth, "CB", nexus.get_data_type(arg_base))
                    cb_idx += 1
                    arg_buf = self.device.create_buffer(arg.base)
                    _, idx = add_arg(arg_buf)
                    # shape flattened
                    for dim in arg.shape:
                        _, idx = add_arg(dim)
                    # strides flattened
                    for stride in arg.strides:
                        _, idx = add_arg(stride)
                    padded = 1  # arg.padding == "nan"
                    _, idx = add_arg(padded)
                    ##  Repeat since the tensor descriptor is lowered with redundant information
                    # shape flattened
                    for dim in arg.shape:
                        _, idx = add_arg(dim)
                    # strides flattened
                    for stride in arg.strides:
                        _, idx = add_arg(stride)
                    # block shape? Not used by kernel
                    buffers.append(arg_buf)
                else:
                    _, idx = add_arg(arg)

            command.finalize([gridX, gridY, gridZ], [num_warps, 1, 1], shared_memory)

        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        self.schedule.run()
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class TTRTUtils(object):

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super(TTRTUtils, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._device = None  # lazy init

    def _init_device(self):
        if self._device is None:
            import ttrt.runtime
            mesh_options = ttrt.runtime.MeshDeviceOptions()
            self._device = ttrt.runtime.open_mesh_device(mesh_options)
            import atexit
            atexit.register(lambda: ttrt.runtime.close_mesh_device(self._device))
        return self._device

    def load_binary(self, name, kernel, shared_mem, device):
        import ttrt
        from ttrt.runtime._ttmlir_runtime.binary import load_binary_from_bytes
        binary = load_binary_from_bytes(kernel)
        # TODO we can probably eliminate this line
        ttrt.runtime.set_compatible_device_runtime(binary.fbb if hasattr(binary, "fbb") else binary)
        self._init_device()
        function = (binary, 0)  # program_index 0 — extend if multi-program later
        return (binary, function, 0, 0, 1)  # module, function, n_regs, n_spills, n_max_threads

    def get_device_properties(self, *args):
        core_count = 130  # ttrt query?
        return {
            "max_num_regs": core_count * 4, "max_shared_mem": 1024 * 1024 * 1024, "multiprocessor_count": core_count,
            "warpSize": 1
        }


class TTRTLauncher(object):

    def __init__(self, src, metadata):
        self.metadata = metadata
        self.signature = {idx: value for idx, value in src.signature.items()}

    def _torch_to_ttrt_dtype(self, dtype):
        import torch, ttrt.runtime
        return {
            torch.float32: ttrt.runtime.DataType.Float32,
            torch.float16: ttrt.runtime.DataType.Float16,
            torch.bfloat16: ttrt.runtime.DataType.BFloat16,
            torch.int32: ttrt.runtime.DataType.Int32,
            torch.uint32: ttrt.runtime.DataType.UInt32,
        }[dtype]

    def _to_runtime_input(self, arg):
        """Wrap any triton arg (tensor / TensorDescriptor / scalar) as a ttrt borrowed
        tensor. Returns (runtime tensor, torch tensor to keep alive, is_tensor_arg).

        is_tensor_arg is True for actual tensors (real torch tensors / TensorDescriptors)
        and False for wrapped scalars — used downstream to decide which slots are
        candidate output buffers.
        """
        import ttrt.runtime
        import torch

        _TORCH_TO_TTRT_DTYPE = {
            torch.float32: ttrt.runtime.DataType.Float32,
            torch.bfloat16: ttrt.runtime.DataType.BFloat16,
            torch.int32: ttrt.runtime.DataType.Int32,
            torch.uint32: ttrt.runtime.DataType.UInt32,
            torch.uint16: ttrt.runtime.DataType.UInt16,
            torch.uint8: ttrt.runtime.DataType.UInt8,
        }

        # Tensor or TensorDescriptor → existing path.
        if hasattr(arg, "base") or isinstance(arg, torch.Tensor):
            t = arg.base if hasattr(arg, "base") else arg
            # if t.dtype in _TORCH_DTYPE_PROMOTION:
            #     t = t.to(_TORCH_DTYPE_PROMOTION[t.dtype]).contiguous()
            rt = ttrt.runtime.create_borrowed_host_tensor(
                t.data_ptr(),
                list(t.shape),
                list(t.stride()),
                t.element_size(),
                _TORCH_TO_TTRT_DTYPE[t.dtype],
            )
            return rt, t, True

        # Python scalar → 1-elem torch buffer. Choose dtype by Python type.
        # TODO: use src.signature[i] from metadata to pick the exact dtype the
        # binary expects (e.g. i32 vs u32 vs fp32). For now, heuristic by type.
        if isinstance(arg, bool):
            scalar_dtype = torch.uint8  # ttrt has no bool; uint8 is the alias
        elif isinstance(arg, int):
            scalar_dtype = torch.int32  # int64 unsupported; truncate
        elif isinstance(arg, float):
            scalar_dtype = torch.float32
        else:
            raise TypeError(f"Don't know how to wrap arg of type {type(arg).__name__}")

        buf = torch.tensor([arg], dtype=scalar_dtype)
        rt = ttrt.runtime.create_borrowed_host_tensor(
            buf.data_ptr(),
            [1],
            [1],
            buf.element_size(),
            _TORCH_TO_TTRT_DTYPE[buf.dtype],
        )
        return rt, buf, False

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        import ttrt.runtime
        import torch
        from ttrt.runtime._ttmlir_runtime.runtime import create_scalar_tensor  # TODO: add to ttrt/runtime/init.py
        binary, program_index = function
        device = TTRTUtils()._device

        sig_types = list(self.signature.values())

        kernel_metadata, launch_metadata, enter_hook, exit_hook = args[:4]
        kernel_args = []
        for i, arg in enumerate(args[4:]):
            ty = sig_types[i]
            if ty == "constexpr":
                continue
            if isinstance(arg, TensorDescriptor):
                base = arg.base
                kernel_args.append(base)
                kernel_args.extend(arg.shape)
                kernel_args.extend(arg.strides)
                kernel_args.append(1 if arg.padding == "nan" else 0)
                kernel_args.extend(arg.shape)  # duplicate (intentional)
                kernel_args.extend(arg.strides)
            else:
                kernel_args.append(arg)

        def _unwrap(a):
            return a.base if hasattr(a, "base") else a

        # wrap inputs and copy to device
        wrapped = [self._to_runtime_input(a) for a in kernel_args]
        host_inputs = [w[0] for w in wrapped]
        # Borrowed host tensors hold raw pointers into the torch buffers — keep the
        # torch tensors alive until submit + wait finish.
        keepalive = [w[1] for w in wrapped]
        is_tensor = [w[2] for w in wrapped]

        host_inputs.append(create_scalar_tensor(0))  # block start
        host_inputs.append(create_scalar_tensor(1))  # block end

        # Push each input to the program's expected on-device layout.
        device_inputs = [
            ttrt.runtime.to_layout(
                host_inputs[i],
                device,
                ttrt.runtime.get_layout(binary, program_index, i),
            ) for i in range(len(host_inputs))
        ]

        # execute
        outputs = ttrt.runtime.submit(device, binary, program_index, device_inputs)
        ttrt.runtime.wait(outputs)

        # outputs[i] corresponds to tensor_args_in_order[i] by positional convention.
        tensor_args_in_order = [ka for ka, is_t in zip(keepalive, is_tensor) if is_t]

        assert len(outputs) <= len(tensor_args_in_order), (
            f"submit returned {len(outputs)} outputs but kernel only has "
            f"{len(tensor_args_in_order)} tensor args")

        output_dst_torch = tensor_args_in_order[-len(outputs):] if outputs else []

        for dst_torch, out_dev in zip(output_dst_torch, outputs):
            host_list = ttrt.runtime.to_host(out_dev, untilize=True, blocking=True)
            promoted = dst_torch.dtype  #_TORCH_DTYPE_PROMOTION.get(dst_torch.dtype, dst_torch.dtype)
            if promoted == dst_torch.dtype:
                ttrt.runtime.memcpy(dst_torch.data_ptr(), host_list[0])
            else:
                intermediate = torch.empty(dst_torch.shape, dtype=promoted)
                ttrt.runtime.memcpy(intermediate.data_ptr(), host_list[0])
                dst_torch.copy_(intermediate)
            ttrt.runtime.deallocate_tensor(out_dev)

        # Clean up device-resident inputs.
        for t in device_inputs:
            ttrt.runtime.deallocate_tensor(t)

        del keepalive


class CPUDriver(DriverBase):

    @staticmethod
    def is_active():
        # Always active so the off-line compiler doesn't complain
        # TODO: Fix the off-line compiler
        return True
        try:
            return bool(CPUDriver.get_device())
        except ImportError:
            return False

    def get_device(self, device_id=0):
        if self.runtime is None:
            self.runtime = get_nexus_runtime()
        return self.runtime.get_device(device_id)

    def __init__(self):
        if (use_ttrt := os.environ.get("TRITON_TTMLIR_TARGET", "")) == "d2m":
            self.utils = TTRTUtils()
            import torch
            self.get_current_stream = lambda idx: torch.cpu.Stream()  # TODO: maybe ttrt/pjrt here?
            self.launcher_cls = TTRTLauncher
        else:
            self.runtime = None
            self.utils = CpuUtils(self)
            import torch
            self.get_current_stream = lambda idx: torch.cpu.Stream()
            self.launcher_cls = CPULauncher

    def get_device_interface(self):
        return CPUDeviceInterface()

    def get_current_device(self):
        return 0

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        capability = "cpu"
        warp_size = 1
        return GPUTarget("tenstorrent", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_empty_device_buffer(self, size, dtype):
        import nexus
        return self.get_device().create_buffer(size, nexus.get_data_type(dtype))

    def get_device_buffer(self, torch_buffer):
        return self.get_device().create_buffer(torch_buffer)

    def copy_buffer_to_host(self, device_buffer, host_buffer):
        return device_buffer.copy(host_buffer)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        cache_size = 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    def clear_cache(self, cache):
        cache.zero_()
