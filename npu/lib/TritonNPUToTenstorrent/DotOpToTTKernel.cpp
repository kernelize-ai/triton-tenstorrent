#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/Support/Debug.h"

#include "Utility.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {

#define DEBUG_TYPE "convert-triton-npu-to-ttkernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct ConvertDotOp : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func.getSymName().ends_with("__compute")) {
      // Hack until we can fix CoreSpecialize
      LDBG("Deleting dot op from non-compute kernel");
      rewriter.replaceOp(op, adaptor.getC());
      return success();
    }

    Location loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    ttkernel::CBType aCBType = cast<ttkernel::CBType>(adaptor.getA().getType());
    ttkernel::CBType bCBType = cast<ttkernel::CBType>(adaptor.getB().getType());

    llvm::errs() << "dot op a cb type: " << aCBType << "\n";
    llvm::errs() << "dot op b cb type: " << bCBType << "\n";

    auto aType = cast<RankedTensorType>(op.getA().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());

    const int64_t aNumTiles = aType.getShape()[0] / 32;
    const int64_t bNumTiles = bType.getShape()[1] / 32;
    // TODO: we should check that aNumTiles * bNumTiles < sizeof(dest)
    llvm::errs() << "dot op a num tiles: " << aNumTiles << "\n";
    llvm::errs() << "dot op b num tiles: " << bNumTiles << "\n";

    Value aNumTilesVal = arith::createConstantI32(
        loc, rewriter, static_cast<int32_t>(aNumTiles));
    Value bNumTilesVal = arith::createConstantI32(
        loc, rewriter, static_cast<int32_t>(bNumTiles));

    ttkernel::CBWaitFrontOp::create(rewriter, loc, adaptor.getA(),
                                    aNumTilesVal);
    ttkernel::CBWaitFrontOp::create(rewriter, loc, adaptor.getB(),
                                    bNumTilesVal);

    // TODO: will the loop carried value break loop unrolling?
    Value destRegisterIndex = arith::createIndexConstant(loc, rewriter, 0);
    scf::ForOp aLoop = scf::ForOp::create(
        rewriter, loc, arith::createConstantI32(loc, rewriter, 0), aNumTilesVal,
        arith::createConstantI32(loc, rewriter, 1),
        ValueRange{destRegisterIndex});
    rewriter.setInsertionPointToStart(aLoop.getBody());
    {
      scf::ForOp bLoop = scf::ForOp::create(
          rewriter, loc, arith::createConstantI32(loc, rewriter, 0),
          bNumTilesVal, arith::createConstantI32(loc, rewriter, 1),
          ValueRange{aLoop.getRegionIterArgs()[0]});
      {
        rewriter.setInsertionPointToStart(bLoop.getBody());

        Value aCBTileIndex = aLoop.getInductionVar();
        Value bCBTileIndex = bLoop.getInductionVar();

        ttkernel::MatmulTilesOp::create(
            rewriter, loc, adaptor.getA(), adaptor.getB(), aCBTileIndex,
            bCBTileIndex, bLoop.getRegionIterArgs()[0]);
        Value nextDestRegisterIndex =
            arith::AddIOp::create(rewriter, loc, bLoop.getRegionIterArgs()[0],
                                  arith::createIndexConstant(loc, rewriter, 1));
        scf::YieldOp::create(rewriter, loc, nextDestRegisterIndex);
      }
      rewriter.setInsertionPointAfter(bLoop);
      scf::YieldOp::create(rewriter, loc, bLoop.getResult(0));
    }

    rewriter.setInsertionPointAfter(aLoop);
#if 1
    Value dst;
    {
      SetVector<Operation *> slice = getSlice(op);

      for (auto *op : slice) {
        llvm::errs() << "slice op: " << *op << "\n";
      }

      auto storeOpIt = llvm::find_if(
          slice, [](Operation *user) { return isa<gpu::LocalStoreOp>(user); });
      assert(storeOpIt != slice.end() &&
             "expected to find local store op in forward slice");
      gpu::LocalStoreOp storeOp = cast<gpu::LocalStoreOp>(*storeOpIt);
      dst = rewriter.getRemappedValue(storeOp.getDst());
    }
    assert(dst && "unable to find output op for dot op");
    auto dstCBType = cast<ttkernel::CBType>(dst.getType());
    llvm::errs() << "dst: " << dst << "\n";
#if 1
    Value numPages =
        arith::createConstantI32(loc, rewriter, dstCBType.getNumTiles());
    ttkernel::CBReserveBackOp::create(rewriter, loc, dst, numPages);
    for (unsigned i = 0; i < dstCBType.getNumTiles(); ++i) {
      ttkernel::PackTileOp::create(
          rewriter, loc, arith::createConstantI32(loc, rewriter, i), dst,
          arith::createConstantI32(loc, rewriter, i));
    }
#else
    // using the loop carried dest counter but really this should be the same as
    // the number of output tiles. do we need it? can we just use a constant
    // instead?
    ttkernel::CBReserveBackOp::create(rewriter, loc, dst, aLoop.getResult(0));
    scf::ForOp packTileLoop = scf::ForOp::create(
        rewriter, loc, arith::createConstantI32(loc, rewriter, 0),
        aLoop.getResult(0), arith::createConstantI32(loc, rewriter, 1),
        ValueRange{});
    rewriter.setInsertionPointToStart(packTileLoop.getBody());
    {
      ttkernel::PackTileOp::create(rewriter, loc,
                                   packTileLoop.getInductionVar(), dst,
                                   packTileLoop.getInductionVar());
    }
    rewriter.setInsertionPointAfter(packTileLoop);
#endif
#endif

    ttkernel::CBPopFrontOp::create(rewriter, loc, adaptor.getA(), aNumTilesVal);
    ttkernel::CBPopFrontOp::create(rewriter, loc, adaptor.getB(), bNumTilesVal);

    // foward the dot op c operand to the users of the dot op
    rewriter.replaceOp(op, adaptor.getC());
    return success();
  }
};

} // namespace

void populateDotOpConversionPattern(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
  patterns.add<ConvertDotOp>(typeConverter, patterns.getContext());
}

} // namespace npu
} // namespace triton
} // namespace mlir
