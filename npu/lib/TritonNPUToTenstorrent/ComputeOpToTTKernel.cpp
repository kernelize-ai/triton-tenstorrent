#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

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

  ConvertBinaryComputeOp(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<npu::tt::BinaryComputeOp>(typeConverter, context) {}

  /// Walk up the stack to find the offset of the register buffer for the given
  /// value.
  uint getRegIndex(Value value) const {
    if (auto op = value.getDefiningOp()) {
      auto helper = tt::TritonTenstorrentDialect::getLoaded(op)
                        ->getAllocOffsetAttrHelper();
      if (helper.isAttrPresent(op)) {
        return helper.getAttr(op).getInt();
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
        if (auto init = loopOp.getTiedLoopInit(blockArg)) {
          return getRegIndex(init->get());
        }
      }
    }
    assert(false && "No allocation offset attribute found");
    return 0;
  }

  LogicalResult
  matchAndRewrite(npu::tt::BinaryComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs =
        arith::createIndexConstant(loc, rewriter, getRegIndex(op.getLhs()));
    Value rhs =
        arith::createIndexConstant(loc, rewriter, getRegIndex(op.getRhs()));
    Value dest = arith::createIndexConstant(loc, rewriter,
                                            getRegIndex(op->getResult(0)));

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
