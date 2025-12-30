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

    def __init__(self):
        runtime = get_nexus_runtime()
        self.device = runtime.get_device(0)

    def load_binary(self, name, kernel, shared_mem, device):
        ## TODO: change to load_library from kernel string so the tmp files are not needed
        ## tmpfile must be persistent since the file will be jit compiled on the first run
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".cpp", delete=False) as f:
            f.write(kernel)
            f.flush()
            os.fsync(f.fileno())
            os.stat(f.name)
            lib = self.device.load_library(f.name)
            kernel = lib.get_kernel(name)
            # TODO: properly handle num registers / max number threads
            return (lib, kernel, 1, shared_mem, 2**12)

    def get_device_properties(self, *args):
        import nexus
        core_count = self.device.get_property_int(nexus.property.Size)
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

        schedule = self.device.create_schedule()
        command = schedule.create_command(function)
        import torch
        ## TODO: Get CB depth from TuningConfig
        cb_depth = 1
        buffers = []
        sig_types = list(self.signature.values())
        idx = 0
        cb_idx = 0
        for arg in args[4:]:
            ty = sig_types[idx]
            if ty == "constexpr":
                continue
            if isinstance(arg, torch.Tensor) or isinstance(arg, nexus.buffer):
                command.set_const(cb_idx, cb_depth, "CB", nexus.get_data_type(arg))
                cb_idx += 1
                arg = self.device.create_buffer(arg)
                buffers.append(arg)
            command.set_arg(idx, arg)
            idx += 1
        command.finalize([gridX, gridY, gridZ], [num_warps, 1, 1], shared_memory)
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        schedule.run()
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


class CPUDriver(DriverBase):

    @staticmethod
    def is_active():
        try:
            return bool(CPUDriver.get_device())
        except ImportError:
            return False

    @staticmethod
    def get_device():
        import nexus
        runtime = get_nexus_runtime()
        device = runtime.get_device(0)
        return device

    def __init__(self):
        self.device = CPUDriver.get_device()
        self.utils = CpuUtils(self.device)
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
        return GPUTarget("tt-metal", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_empty_device_buffer(self, size, dtype):
        import nexus
        return self.device.create_buffer(size, nexus.get_data_type(dtype))

    def get_device_buffer(self, torch_buffer):
        return self.device.create_buffer(torch_buffer)

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
