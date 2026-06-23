#include "npu/include/TritonNPUToD2M/Passes.h"

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir {

using namespace tt;

namespace triton {
namespace npu {

#define GEN_PASS_DEF_TENSORIZESCALARS
#include "npu/include/TritonNPUToD2M/Passes.h.inc"

#define DEBUG_TYPE "tensorize-scalars"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Convert all scalar integer types to 32-bit unsigned integers and wrap them
/// in a 0D ranked tensor with a scalar TTNN layout.
class TensorizeScalarsTypeConverter : public mlir::TypeConverter {
public:
  using mlir::TypeConverter::convertType;

  TensorizeScalarsTypeConverter(mlir::MLIRContext *ctx) : TypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](IntegerType type) -> Type {
      auto ui32Type =
          IntegerType::get(type.getContext(), 32, IntegerType::Unsigned);
      SmallVector<int64_t> gridShape = {1, 1};
      auto bufferTypeAttr = mlir::tt::ttnn::BufferTypeAttr::get(
          type.getContext(), mlir::tt::ttnn::BufferType::SystemMemory);
      auto memrefType = MemRefType::get(
          {}, ui32Type, AffineMap::getMultiDimIdentityMap(0, type.getContext()),
          bufferTypeAttr);
      auto scalarLayout = mlir::tt::ttnn::TTNNLayoutAttr::get(
          type.getContext(),
          AffineMap::getMultiDimIdentityMap(0, type.getContext()), gridShape,
          memrefType,
          /*memLayout=*/mlir::tt::ttnn::TensorMemoryLayoutAttr{},
          /*tensorMesh=*/nullptr,
          /*ignorePhysicalLayout=*/false,
          /*coreRangeSet=*/mlir::tt::ttnn::CoreRangeSetAttr{});
      return RankedTensorType::get({}, ui32Type, scalarLayout);
    });
  };
};

/// Convert the `ttnn.generic` operation that uses the converted function
/// arguments
/// (`TensorizeScalarsTypeConverter`) and have them use the 0D-tensorized
/// version instead.
class ConvertTTNNGenericOp : public OpConversionPattern<mlir::ttnn::GenericOp> {
  using OpConversionPattern<mlir::ttnn::GenericOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::ttnn::GenericOp op,
                  OpConversionPattern<mlir::ttnn::GenericOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LDBG("converting ttnn.generic op");

    SmallVector<std::pair<OpOperand *, Value>> replacements;
    for (auto [original, converted] :
         llvm::zip(op->getOpOperands(), adaptor.getOperands())) {
      if (original.get() != converted) {
        replacements.push_back({&original, converted});
      }
    }

    if (!replacements.empty()) {
      rewriter.modifyOpInPlace(op, [&]() {
        for (auto [original, converted] : replacements) {
          LDBG("replacing ttnn.generic operand " << original->get() << " with "
                                                 << converted);
          op->setOperand(original->getOperandNumber(), converted);
        }
      });
    }

    return success();
  }
};

} // namespace

struct TensorizeScalarsPass
    : public impl::TensorizeScalarsBase<TensorizeScalarsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TensorizeScalarsTypeConverter typeConverter(context);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      // Ensure all `func.func` operations have tensorized scalars, see
      // `populateFunctionOpInterface...`.
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<mlir::ttnn::GenericOp>(
        [&](mlir::ttnn::GenericOp op) {
          // Ensure `ttnn.generic` operations have replaced their uses of
          // `func.func` arguments, see `ConvertTTNNGenericOp`.
          bool hasDanglingScalar =
              llvm::any_of(op.getOperandTypes(), [](Type type) {
                return mlir::isa<IntegerType>(type);
              });
          return !hasDanglingScalar;
        });
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *op) {
      // Ensure all other operations are legal, since we only care about
      // `func.func` and `ttnn.generic` operations.
      return true;
    });

    mlir::RewritePatternSet patterns(context);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    patterns.add<ConvertTTNNGenericOp>(typeConverter, patterns.getContext());
    if (failed(applyFullConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
