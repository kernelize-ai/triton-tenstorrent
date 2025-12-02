#ifndef TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_PATTERNS_H
#define TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_PATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
namespace npu {

class PointerInfoAnalysis;

void populateFuncOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

void populateComputeOpConversionPattern(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

void populateElementwiseOpConversionPattern(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit);

void populateMakeRangeOpConversionPattern(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit);

void populateMemoryOpConversionPattern(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PointerInfoAnalysis *pointerInfoAnalysis,
                                       PatternBenefit benefit);

void populateSPMDOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

void populateViewOpConversionPattern(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

} // namespace npu
} // namespace triton
} // namespace mlir

#endif // TRITON_NPU_CONVERSION_TRITONNPU_TO_TENSTORRENT_PATTERNS_H
