#ifndef TRITON_DIALECT_TRITONTENSTORRENT_IR_ATTRIBUTES_H_
#define TRITON_DIALECT_TRITONTENSTORRENT_IR_ATTRIBUTES_H_

#include "mlir/IR/Attributes.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_ATTRDEF_CLASSES
#include "npu/include/Dialect/TritonTenstorrent/IR/Enums.h.inc"

#include "npu/include/Dialect/TritonTenstorrent/IR/AttrDefs.h.inc"

#endif // TRITON_DIALECT_TRITONTENSTORRENT_IR_ATTRIBUTES_H_
