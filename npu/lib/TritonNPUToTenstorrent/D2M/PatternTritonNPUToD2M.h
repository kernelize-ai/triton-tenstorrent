#ifndef TRITON_NPU_CONVERSION_TRITONNPU_TO_D2M_PATTERNS_H
#define TRITON_NPU_CONVERSION_TRITONNPU_TO_D2M_PATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
namespace npu {
namespace experimental {

void populateFuncOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

}
} // namespace npu
} // namespace triton
} // namespace mlir

#endif
