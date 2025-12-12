#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"

#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace npu {

#define GEN_PASS_DEF_TRITONTENSTORRENTCONVERTTENSORDESCTOLOADSTORE
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritontenstorrent-convert-tensor-desc"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

// TODO: we may have to structure this like a conversion pass so we can use a
// ConversionTarget to rewrite TensorDesc to ptr.

} // namespace

class TritonTenstorrentConvertTensorDescPass
    : public npu::impl::TritonTenstorrentConvertTensorDescToLoadStoreBase<
          TritonTenstorrentConvertTensorDescPass> {
public:
  void runOnOperation() override {
    auto op = getOperation();

    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                      mlir::scf::SCFDialect,
                                      mlir::triton::TritonDialect>(
        [](mlir::Operation *op) {
          return !hasATensorDescriptorType(op->getOperandTypes()) &&
                 !hasATensorDescriptorType(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
      return !hasATensorDescriptorType(funcOp.getFunctionType().getInputs()) &&
             !hasATensorDescriptorType(funcOp.getFunctionType().getResults());
    });

    mlir::TypeConverter converter;

    converter.addConversion([](mlir::Type t) {
      // Most types don't require any conversion
      return t;
    });
    converter.addConversion([](mlir::triton::TensorDescType t,
                               llvm::SmallVectorImpl<mlir::Type> &out) {
      // We convert a tensor descriptor into an pointer, and a shape and stride
      // for each dimension, and padding option. i.e., we create 1+2*rank+1
      // values. Note that tensor descriptors may be signed/unsigned integers
      // whereas pointers should always be signless.
      auto tensorType = t.getSignlessBlockType();
      out.push_back(triton::getPointerType(tensorType.getElementType()));
      out.insert(out.end(), 2 * tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 64));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      return mlir::success();
    });

    mlir::RewritePatternSet patterns(op->getContext());

    // Populate conversion patterns to handle loops, function calls, and arith
    // ops.
    triton::populateFunctionTypeConversions(converter, patterns);
    mlir::scf::populateSCFStructuralTypeConversions(converter, patterns);
    triton::populateArithTypeConversions(converter, patterns);

    assert(false && "TODO");

    ConversionConfig config;
    config.buildMaterializations = false;

    if (mlir::failed(mlir::applyPartialConversion(
            op, target, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace npu
} // namespace triton
} // namespace mlir
