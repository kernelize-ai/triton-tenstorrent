#ifndef TRITON_DIALECT_TRITONTENSTORRENT_IR_ATTRIBUTES_H_
#define TRITON_DIALECT_TRITONTENSTORRENT_IR_ATTRIBUTES_H_

#include "mlir/IR/Attributes.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

inline constexpr llvm::StringLiteral kIOTypeAttrName =
    "triton_tenstorrent.io_type";

#define GET_ATTRDEF_CLASSES
#include "npu/include/Dialect/TritonTenstorrent/IR/AttrDefs.h.inc"

#endif // TRITON_DIALECT_TRITONTENSTORRENT_IR_ATTRIBUTES_H_
