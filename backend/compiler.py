from importlib.metadata import metadata
import tempfile
import functools
import hashlib
import re
import os
from typing import Tuple
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, cpu
from triton import knobs
from triton.runtime.build import _build
import triton.backends.cpu.driver as cpu_driver

from dataclasses import dataclass
from typing import Dict
from types import ModuleType


@dataclass(frozen=True)
class CPUOptions:
    num_warps: int = int(os.environ.get('TRITON_CPU_NUM_WARPS', 1))
    num_ctas: int = 1
    num_stages: int = 1
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    arch: str = None
    enable_fp_fusion: bool = True
    backend_name: str = 'cpu'
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    matrix_instr_nonkdim: int = 16
    warp_size: int = 1
    min_dot_size: int = 1

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    instrumentation = None  # TODO: intra-kernel instrumentation not yet supported

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "tenstorrent"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "so"
        self.device = 'Tenstorrent'

    def parse_options(self, options):
        args = {'arch': cpu.get_processor_name()}
        if "enable_fp_fusion" not in options:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        args.update(
            {k: options[k]
             for k in CPUOptions.__dataclass_fields__.keys()
             if k in options if options[k] is not None})

        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        return dict()

    def get_module_map(self) -> Dict[str, ModuleType]:
        # TODO
        return {"triton.language.extra.libdevice": None}

    def load_dialects(self, ctx):
        cpu.load_dialects(ctx, self.device)

    @staticmethod
    def parse_attr(desc):
        ret = []
        if "D" in desc:
            ret += [["tt.divisibility", 8]]
        # pop D from desc
        desc = desc.replace("D", "")
        ret += BaseBackend.parse_attr(desc)
        return ret

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        threads_per_warp = 1
        metadata["warp_size"] = threads_per_warp
        num_ctas = 1
        passes.ttir.add_convert_to_ttgpuir(pm, "cpu", options.num_warps, threads_per_warp, num_ctas)
        cpu.passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, 'make_ttgir')

        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # Triton -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.convert.add_scf_to_cf(pm)
        cpu.passes.ttcpuir.add_allocate_shared_memory(pm)
        passes.convert.add_index_to_llvmir(pm)

        cpu.passes.ttcpuir.add_to_llvmir(pm)
        cpu.passes.ttcpuir.add_masked_ops_to_llvm(pm)
        cpu.passes.ttcpuir.add_shared_memory_global_conversion(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, 'make_llir')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        cpu.attach_target_triple(llvm_mod, cpu.get_default_target_triple())
        target_features = ''
        llvm.attach_datalayout(llvm_mod, cpu.get_default_target_triple(), options.arch, target_features)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)
        metadata["shared"] = src.get_int_attr("ttg.shared")

        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_tenstorrent_mlir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        cpu.passes.tenstorrent.add_convert_compute_ops(pm)
        cpu.passes.tenstorrent.add_propagate_register_indices(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        cpu.passes.tenstorrent.add_core_specialize(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)

        cpu.passes.tenstorrent.add_to_ttkernel_dialect(pm)
        passes.common.add_canonicalizer(pm)

        cpu.passes.tenstorrent.add_finalize_cb_transactions(pm)

        # tt-mlir pipeline
        cpu.passes.tenstorrent.add_ttkernel_control_dst_selection(pm)

        pm.run(mod, "make_ttmlir")
        return mod

    @staticmethod
    def make_ttmlir_cpp_file(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # tt-mlir continued
        cpu.passes.tenstorrent.add_ttkernel_device_zone_scopes(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_licm(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)

        cpu.passes.common.add_arith_int_range_opts(pm)
        cpu.passes.tenstorrent.add_ttkernel_to_emitc(pm)
        passes.common.add_canonicalizer(pm)
        cpu.passes.tenstorrent.add_form_expressions_pass(pm)

        pm.run(mod, "make_ttmlir_cpp")

        # find function names
        src = str(mod)
        names = re.findall(r"func.func public @(?!(?:barrier)\b)([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 3
        compute_kernel = next((name for name in names if name.endswith("__compute")), None)
        reader_kernel = next((name for name in names if name.endswith("__reader")), None)
        writer_kernel = next((name for name in names if name.endswith("__writer")), None)
        assert compute_kernel is not None and reader_kernel is not None and writer_kernel is not None

        metadata["name"] = compute_kernel[:-9]  # remove __compute suffix
        metadata["shared"] = 0  # TODO: store cb sizes in module attributes?
        metadata["profile_scratch_size"] = 0
        metadata["profile_scratch_align"] = 1

        cpp_file = "#ifdef COMPUTE_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, compute_kernel)
        cpp_file += "\n#endif  // COMPUTE_KERNEL\n\n"
        cpp_file += "\n#ifdef READER_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, reader_kernel)
        cpp_file += "\n#endif  // READER_KERNEL\n\n"
        cpp_file += "\n#ifdef WRITER_KERNEL\n"
        cpp_file += cpu.translate_to_cpp(mod, writer_kernel)
        cpp_file += "\n#endif  // WRITER_KERNEL\n"

        return cpp_file

    @staticmethod
    def make_asm(src, metadata, options):
        names = re.findall(r"define void @(?!(?:barrier)\b)([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]

        flags = []
        return llvm.translate_to_asm(src, cpu.get_default_target_triple(), options.arch, '', flags,
                                     options.enable_fp_fusion, False)

    @staticmethod
    def make_library(src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            Path(asm_path).write_text(src)
            lib_dirs = cpu_driver.library_dirs()
            libs = ["sleef", "cpu_utils"]  # TODO: conditionally include?
            include_dirs = []
            ccflags = []
            for lib_dir in lib_dirs:
                ccflags.extend(["-Xlinker", "-rpath", "-Xlinker", lib_dir])
            if cpu_driver.is_macos():
                ccflags.extend(["-undefined", "dynamic_lookup"])
            so = _build("kernel", asm_path, tmpdir, lib_dirs, include_dirs, libs, ccflags)
            with open(so, "rb") as f:
                return f.read()

    @staticmethod
    def make_tenstorrent_binary(src, metadata, options):
        # TODO: Implement the Tenstorrent binary generation
        pass

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        elif language == Language.GLUON:
            raise NotImplementedError("Gluon language support is not implemented for NPU backend")

        if self.device == 'CPU':
            stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
            stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
            stages["so"] = lambda src, metadata: self.make_library(src, metadata, options)
        elif self.device == 'Tenstorrent':
            stages["ttmlir"] = lambda src, metadata: self.make_tenstorrent_mlir(src, metadata, options)
            stages["cpp"] = lambda src, metadata: self.make_ttmlir_cpp_file(src, metadata, options)
            stages['so'] = lambda src, metadata: self.make_tenstorrent_binary(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        version = 0.1
        return f'{version}-{self.target.arch}'
