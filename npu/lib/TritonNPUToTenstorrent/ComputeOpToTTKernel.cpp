#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "Utility.h"

namespace mlir {
using namespace tt;

namespace triton {
namespace npu {

namespace {

struct ConvertBinaryComputeOp
    : public OpConversionPattern<npu::tt::BinaryComputeOp> {
  using OpConversionPattern<npu::tt::BinaryComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(npu::tt::BinaryComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto lhsReg = lookupRegisterIndex(op.getLhs());
    auto rhsReg = lookupRegisterIndex(op.getRhs());
    auto destReg = lookupRegisterIndex(op->getResult(0));
    Value lhs = arith::createIndexConstant(loc, rewriter, lhsReg);
    Value rhs = arith::createIndexConstant(loc, rewriter, rhsReg);
    Value dest = arith::createIndexConstant(loc, rewriter, destReg);

    std::string opcode = op.getOpcode().str();
    if (opcode == "arith.addf") {
      ttkernel::AddBinaryTilesInitOp::create(rewriter, loc);
      ttkernel::AddBinaryTilesOp::create(rewriter, loc, lhs, rhs, dest);
    } else if (opcode == "arith.subf") {
      ttkernel::SubBinaryTilesInitOp::create(rewriter, loc);
      ttkernel::SubBinaryTilesOp::create(rewriter, loc, lhs, rhs, dest);
    } else if (opcode == "arith.mulf") {
      ttkernel::MulBinaryTilesInitOp::create(rewriter, loc);
      ttkernel::MulBinaryTilesOp::create(rewriter, loc, lhs, rhs, dest);
    } else if (opcode == "arith.divf") {
      ttkernel::DivBinaryTilesInitOp::create(rewriter, loc);
      ttkernel::DivBinaryTilesOp::create(rewriter, loc, lhs, rhs, dest);
    } else {
      // LDBG("Unsupported opcode: " << opcode.c_str());
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateComputeOpConversionPattern(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  patterns.add<ConvertBinaryComputeOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
