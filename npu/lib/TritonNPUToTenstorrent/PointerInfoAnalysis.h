#ifndef TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_POINTERINFOANALYSIS_H
#define TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_POINTERINFOANALYSIS_H

#include <optional>

#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace npu {

struct PointerInfo {
  mlir::Value basePtr;
};

class PointerInfoAnalysis {
public:
  explicit PointerInfoAnalysis(mlir::Operation *root);

  std::optional<PointerInfo> getInfo(mlir::triton::LoadOp loadOp) const {
    auto it = loadInfo.find(loadOp.getOperation());
    if (it != loadInfo.end()) {
      return it->second;
    }
    return std::nullopt;
  }

private:
  mlir::DenseMap<Operation *, PointerInfo> loadInfo;
  mlir::DenseMap<Value, std::optional<PointerInfo>> cache;

#if 0
    std::optional<PointerInfo> traceValue(mlir::Value v);
    std::optional<PointerInfo> traceBlockArgument(mlir::BlockArgument arg);
#endif
};

} // namespace npu
} // namespace triton
} // namespace mlir

#endif // TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_POINTERINFOANALYSIS_H
