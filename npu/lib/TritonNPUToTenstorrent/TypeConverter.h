#ifndef TRITON_CONVERSION_TRITONNPU_TO_TENSTORRENT_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONNPU_TO_TENSTORRENT_TYPECONVERTER_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
namespace npu {

class TritonNPUToTenstorrentTypeConverter : public mlir::TypeConverter {
public:
  using mlir::TypeConverter::convertType;

  TritonNPUToTenstorrentTypeConverter(mlir::MLIRContext *ctx);
};

} // namespace npu
} // namespace triton
} // namespace mlir

#endif
