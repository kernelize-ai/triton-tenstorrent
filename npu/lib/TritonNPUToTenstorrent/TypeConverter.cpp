#include "TypeConverter.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

TritonNPUToTenstorrentTypeConverter::TritonNPUToTenstorrentTypeConverter(
    MLIRContext *ctx) {
  addConversion([](Type type) { return type; });

  addConversion([&](RankedTensorType tensorTy) -> Type {
    Type elemTy = tensorTy.getElementType();
    if (isa<triton::PointerType>(elemTy)) {
      elemTy = IntegerType::get(tensorTy.getContext(), 64);
    }
    return VectorType::get(tensorTy.getShape(), elemTy);
  });
}
