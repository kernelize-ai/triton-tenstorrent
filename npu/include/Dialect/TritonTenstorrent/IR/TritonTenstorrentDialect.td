#ifndef TRITONTENSTORRENT_DIALECT
#define TRITONTENSTORRENT_DIALECT

include "mlir/IR/OpBase.td"

def TritonTenstorrent_Dialect : Dialect {
  let name = "triton_tenstorrent";

  let cppNamespace = "::mlir::triton::npu::tt";

  let description = [{
    Triton Tenstorrent Dialect. Helpers ops for progressive lowering from TritonGPU IR to Tenstorrent TTKernel IR.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
  ];

  let discardableAttrs = (ins
    // Derived ttkernel func attributes
    "::mlir::StringAttr":$initial_func,
    "::mlir::triton::npu::tt::DerivedFuncTypeAttr":$func_type,
    "::mlir::DenseI32ArrayAttr":$cb_tile_sizes,
    // Index of allocated CB
    "::mlir::IntegerAttr":$alloc_index,
    // Register Offset and Size
    "::mlir::IntegerAttr":$reg_offset,
    "::mlir::IntegerAttr":$reg_size
  );

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
