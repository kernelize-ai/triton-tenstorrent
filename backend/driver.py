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


class TTUtils(object):

    def __init__(self, driver):
        self.driver = driver

    def load_binary(self, name, kernel, shared_mem, device_id):
        device = self.driver.get_device(device_id)
        lib = device.load_library(kernel, len(kernel))
        kernel = lib.get_kernel(name)
        return (lib, kernel, 1, shared_mem, 2**12)

    def get_device_properties(self, *args):
        import nexus
        core_count = self.driver.get_device().get_property_int(nexus.property.Size)
        return {
            "max_num_regs": core_count * 4, "max_shared_mem": 1024 * 1024 * 1024, "multiprocessor_count": core_count,
            "warpSize": 1
        }

class TTLauncher(object):

    def __init__(self, src, metadata):
        import torch
        import torch_nexus
        self.device = torch.nexus.get_device()
        self.schedule = None
        self.command = None
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        self.constants = {arg_idx(idx): value for idx, value in constants.items()}
        self.signature = {idx: value for idx, value in src.signature.items()}

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        import nexus
        import torch

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

        # TODO: also check for changes to launch parameters
        #if self.schedule is None:
        self.schedule = self.device.create_schedule()
        self.command = self.schedule.create_command(function)
        ## TODO: Get CB depth from TuningConfig
        cb_depth = 8
        sig_types = list(self.signature.values())
        idx = 0
        def add_arg(arg):
            nonlocal idx
            self.command.set_arg(idx, arg)
            idx += 1
        cb_idx = 0
        def add_const(arg):
            nonlocal cb_idx
            self.command.set_const(cb_idx, cb_depth, "CB", nexus.get_data_type(arg))
            cb_idx += 1

        for i, arg in enumerate(args[4:]):
            ty = sig_types[i]
            if ty == "constexpr":
                continue
            if isinstance(arg, torch.Tensor) or isinstance(arg, nexus.Buffer):
                add_const(arg)
                add_arg(arg)
            elif isinstance(arg, TensorDescriptor):
                arg_base = arg.base
                add_const(arg_base)
                add_arg(arg_base)
                # shape flattened
                for dim in arg.shape:
                    add_arg(dim)
                # strides flattened
                for stride in arg.strides:
                    add_arg(stride)

                # padding
                add_arg(1)
                # shape flattened
                for dim in arg.shape:
                    add_arg(dim)
                # strides flattened
                for stride in arg.strides:
                    add_arg(stride)

            else:
                add_arg(arg)

        self.command.finalize([gridX, gridY, gridZ], [num_warps, 1, 1], shared_memory)

        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        self.schedule.run()
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)

class TTDeviceInterface:

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
            return TTDeviceInterface.HooksTimeAccessor(self)
        return TTDeviceInterface.TimerEvent()

class TTDriver(DriverBase):

    @staticmethod
    def is_active():
        # Always active so the off-line compiler doesn't complain
        try:
            import torch
            import torch_nexus
            return bool(torch.nexus.set_runtime("tt-metal"))
        except Exception as e:
            print("Exception")
            print(e)
            return False

    def __init__(self):
        import torch
        import torch_nexus
        self.device = torch.nexus.set_runtime("tt-metal")
        #self.stream = torch.nexus.get_stream()
        self.torch_device = torch.nexus
        #self.get_current_stream = 0
        self.utils = TTUtils(self)
        self.launcher_cls = TTLauncher
        self.get_device = self.torch_device.get_device
        self.set_current_device = self.torch_device.set_device
        self.get_current_device = self.torch_device.current_device
        self.get_current_stream = lambda idx: torch.cpu.Stream()

    def get_device_interface(self):
        return TTDeviceInterface()

    def get_current_target(self):
        capability = "cpu"
        warp_size = 1
        return GPUTarget("tenstorrent", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("nexus", self.get_current_device())

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        cache_size = 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    def clear_cache(self, cache):
        cache.zero_()
