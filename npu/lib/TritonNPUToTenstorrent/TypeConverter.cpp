#include "TypeConverter.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

using namespace mlir;
using namespace mlir::triton;

TritonNPUToTenstorrentTypeConverter::TritonNPUToTenstorrentTypeConverter(
    MLIRContext *ctx) {
  addConversion([](Type type) { return type; });

  addConversion([&](RankedTensorType tensorTy) -> Type {
  // TODO: introduce an encoding for circular buffer tiles and convert that to
  // ttkernel.cb type
#if 0
    if (auto cbEnc = dyn_cast<>(tensorTy.getEncoding())) {
      // TODO: convert this
    }
#endif
    return tensorTy;
  });
}
