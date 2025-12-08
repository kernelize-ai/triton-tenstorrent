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

  std::optional<PointerInfo> getInfo(Operation *op) const {
    auto it = ptrInfo.find(op);
    if (it != ptrInfo.end()) {
      return it->second;
    }
    return std::nullopt;
  }

private:
  mlir::DenseMap<Operation *, PointerInfo> ptrInfo;
};

} // namespace npu
} // namespace triton
} // namespace mlir

#endif // TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_POINTERINFOANALYSIS_H
