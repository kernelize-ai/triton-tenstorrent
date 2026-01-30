#include "PatternTritonNPUToTenstorrent.h"

#include "mlir/Transforms/DialectConversion.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Utility.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

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

  // TODO: adjust based on dest accumulation mode and/or double buffering mode
  static constexpr unsigned kMaxMatmulDestRegisters = 8; 

  std::pair<unsigned, unsigned> computeSubBlocksSizes(unsigned mTiles,
                                                        unsigned nTiles,
                                                        unsigned availableRegs) const {
    unsigned mSubBlocks = 1;
    unsigned nSubBlocks = 1;

    while ((mTiles / mSubBlocks) * (nTiles / nSubBlocks) > availableRegs) {
      if ((mTiles / mSubBlocks) >= (nTiles / nSubBlocks)) {
        mSubBlocks++;
      } else {
        nSubBlocks++;
      }
    }

    return {mSubBlocks, nSubBlocks};
  }

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
    auto dialect =
        op->getContext()->getLoadedDialect<tt::TritonTenstorrentDialect>();
    int64_t alloc_offset =
        dialect->getAllocOffsetAttrHelper().getAttr(op).getInt();
    int64_t alloc_size = dialect->getAllocSizeAttrHelper().getAttr(op).getInt();
    LDBG("Dot op allocation offset: " << alloc_offset
                                      << " size: " << alloc_size);

    auto typeConverter = getTypeConverter();

    ttkernel::CBType aCBType = cast<ttkernel::CBType>(adaptor.getA().getType());
    ttkernel::CBType bCBType = cast<ttkernel::CBType>(adaptor.getB().getType());

    auto aType = cast<RankedTensorType>(op.getA().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());
    if (aType.getRank() != 2 || bType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2D dot");

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    int64_t mTiles = (M + kTileDimSize - 1) / kTileDimSize;
    int64_t kTiles = (K + kTileDimSize - 1) / kTileDimSize;
    int64_t nTiles = (N + kTileDimSize - 1) / kTileDimSize;

    assert(mTiles * kTiles == aCBType.getNumTiles() &&
           "a cb num tiles does not match a tensor shape");
    assert(kTiles * nTiles == bCBType.getNumTiles() &&
           "b cb num tiles does not match b tensor shape");
    assert(mTiles * nTiles == alloc_size &&
           "output size does not match output tensor shape");

    Value aNumInputTilesValue = arith::createConstantI32(
        loc, rewriter, static_cast<int32_t>(aCBType.getNumTiles()));
    Value bNumInputTilesValue = arith::createConstantI32(
        loc, rewriter, static_cast<int32_t>(bCBType.getNumTiles()));

    ttkernel::CBWaitFrontOp::create(rewriter, loc, adaptor.getA(),
                                    aNumInputTilesValue);
    ttkernel::CBWaitFrontOp::create(rewriter, loc, adaptor.getB(),
                                    bNumInputTilesValue);

    Value zeroI32 = arith::createConstantI32(loc, rewriter, 0);
    Value oneI32 = arith::createConstantI32(loc, rewriter, 1);

    // We need at least 1 register available for the matmul destination
    assert(alloc_offset < kMaxMatmulDestRegisters - 1 &&
           "not enough registers available for matmul");
    unsigned availableRegs = kMaxMatmulDestRegisters - alloc_offset;
    
    auto [mSubBlocks, nSubBlocks] = computeSubBlocksSizes(
        static_cast<unsigned>(mTiles), static_cast<unsigned>(nTiles),
        availableRegs);
    LDBG("DotOp matmul tiles with mSubBlocks=" << mSubBlocks
                                               << " nSubBlocks=" << nSubBlocks);
#if 1
    for (unsigned m = 0; m < static_cast<unsigned>(mTiles); ++m) {
      for (unsigned n = 0; n < static_cast<unsigned>(nTiles); ++n) {
        const unsigned destRegisterIndex = alloc_offset + m * static_cast<unsigned>(nTiles) + n;
        Value destRegisterIndexValue = arith::createIndexConstant(
            loc, rewriter, destRegisterIndex);
        llvm::errs() << "DotOp matmul tiles for m=" << m << " n=" << n
                     << " dest reg index=" << destRegisterIndex << "\n";
        for (unsigned k = 0; k < static_cast<unsigned>(kTiles); ++k) {
          Value kIv = arith::createIndexConstant(loc, rewriter, k);
  
          // aTile = m * kTiles + k
          Value aTile = arith::createIndexConstant(
              loc, rewriter, m * static_cast<unsigned>(kTiles) + k);

          // bTile = k * nTiles + n
          Value bTile = arith::createIndexConstant(
              loc, rewriter, k * static_cast<unsigned>(nTiles) + n);

          ttkernel::MatmulTilesOp::create(rewriter, loc, adaptor.getA(),
                                          adaptor.getB(), aTile, bTile,
                                          destRegisterIndexValue);
        }
      }
    }
#else
   Value mTilesVal =
        arith::createConstantI32(loc, rewriter, static_cast<int32_t>(mTiles));
    Value nTilesVal =
        arith::createConstantI32(loc, rewriter, static_cast<int32_t>(nTiles));
    Value kTilesVal =
        arith::createConstantI32(loc, rewriter, static_cast<int32_t>(kTiles));

    Value destRegisterInitial =
        arith::createIndexConstant(loc, rewriter, alloc_offset);
    auto mLoop = scf::ForOp::create(rewriter, loc, zeroI32, mTilesVal, oneI32,
                                    ValueRange{destRegisterInitial});
    {
      rewriter.setInsertionPointToStart(mLoop.getBody());
      Value mIv = mLoop.getInductionVar();
      Value destRegisterIndexM = mLoop.getRegionIterArgs()[0];

      auto nLoop = scf::ForOp::create(rewriter, loc, zeroI32, nTilesVal, oneI32,
                                      ValueRange{destRegisterIndexM});
      {
        rewriter.setInsertionPointToStart(nLoop.getBody());

        Value nIv = nLoop.getInductionVar();
        Value destRegisterIndexN = nLoop.getRegionIterArgs()[0];

        auto kLoop =
            scf::ForOp::create(rewriter, loc, zeroI32, kTilesVal, oneI32);
        {
          rewriter.setInsertionPointToStart(kLoop.getBody());
          Value kIv = kLoop.getInductionVar();

          // aTile = m * kTiles + k
          Value aTile = arith::AddIOp::create(
              rewriter, loc,
              (arith::MulIOp::create(rewriter, loc, mIv, kTilesVal)), kIv);

          // bTile = k * nTiles + n
          Value bTile = arith::AddIOp::create(
              rewriter, loc,
              (arith::MulIOp::create(rewriter, loc, kIv, nTilesVal)), nIv);

          ttkernel::MatmulTilesOp::create(rewriter, loc, adaptor.getA(),
                                          adaptor.getB(), aTile, bTile,
                                          destRegisterIndexN);
          // scf::YieldOp::create(rewriter, loc);
        }
        rewriter.setInsertionPointAfter(kLoop);
        Value nextDestRegisterIndex =
            arith::AddIOp::create(rewriter, loc, destRegisterIndexN,
                                  arith::createIndexConstant(loc, rewriter, 1));
        scf::YieldOp::create(rewriter, loc, nextDestRegisterIndex);
      }
      rewriter.setInsertionPointAfter(nLoop);
      scf::YieldOp::create(rewriter, loc, nLoop.getResult(0));
    }
    rewriter.setInsertionPointAfter(mLoop);
#endif 

    ttkernel::CBPopFrontOp::create(rewriter, loc, adaptor.getA(),
                                   aNumInputTilesValue);
    ttkernel::CBPopFrontOp::create(rewriter, loc, adaptor.getB(),
                                   bNumInputTilesValue);

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
