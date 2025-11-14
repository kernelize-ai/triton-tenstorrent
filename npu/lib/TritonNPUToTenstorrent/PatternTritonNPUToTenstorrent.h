#ifndef TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_PATTERNS_H
#define TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_PATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
namespace npu {

void populateFuncOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

void populateComputeOpConversionPattern(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

} // namespace npu
} // namespace triton
} // namespace mlir

#endif // TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_PATTERNS_H
