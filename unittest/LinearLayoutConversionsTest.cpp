#include "mlir/IR/MLIRContext.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

namespace mlir::triton::npu {
namespace {

class LinearLayoutConversionsTest : public ::testing::Test {
   public:
   void SetUp() {
    ctx.getOrLoadDialect<tt::TritonTenstorrentDialect>();
    ctx.getOrLoadDialect<triton::gpu::TritonGPUDialect>();
   }

   tt::TiledEncodingAttr tiled(ArrayRef<unsigned> tilesPerCore,
                              ArrayRef<unsigned> tileShape,
                              ArrayRef<unsigned> order) {
    return tt::TiledEncodingAttr::get(&ctx, tilesPerCore,
                                     order,
                                      tileShape);
   }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

void applyLayoutAndPrint(LinearLayout& layout, ArrayRef<std::pair<StringAttr, int32_t>> ins) {
    auto out = layout.apply(ins);
    for (unsigned i = 0; i < ins.size(); i++) {
        llvm::errs() << ins[i].second;
        if (i + 1 < ins.size())
            llvm::errs() << ", ";
    }
    llvm::errs() << " = ";
    for (unsigned i = 0; i < out.size(); i++) {
        llvm::errs() << out[i].second;
        if (i + 1 < out.size())
            llvm::errs() << ", ";
    }
    llvm::errs() << "\n";
}

TEST_F(LinearLayoutConversionsTest, Tiled_OneTile) {
    auto enc = tiled({1, 1}, {32, 32}, {1, 0});
    llvm::errs() << "enc = " << enc << "\n";
    auto layout = toLinearLayout({32, 32}, enc);
    llvm::errs() << "layout = " << layout << "\n";

#if 0
    applyLayoutAndPrint(layout, {{S("register"), 0}, {S("tile"), 0}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 16}, {S("tile"), 0}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 31}, {S("tile"), 0}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 32}, {S("tile"), 0}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 31}, {S("tile"), 1}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 31}, {S("tile"), 0}, {S("block"), 1}});
#endif 

    EXPECT_EQ(layout, LinearLayout({
        {S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
        {S("tile"), {}},
        {S("block"), {}},
    }, {S("dim0"), S("dim1")}));
    
}

TEST_F(LinearLayoutConversionsTest, Tiled_MultiTile) {
    auto enc = tiled({2, 2}, {32, 32}, {1, 0});
    llvm::errs() << "enc = " << enc << "\n";
    auto layout = toLinearLayout({64, 64}, enc);
    llvm::errs() << "layout = " << layout << "\n";

#if 1
    applyLayoutAndPrint(layout, {{S("register"), 0}, {S("tile"), 0}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 16}, {S("tile"), 0}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 31}, {S("tile"), 0}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 32}, {S("tile"), 0}, {S("block"), 0}});

    applyLayoutAndPrint(layout, {{S("register"), 0}, {S("tile"), 1}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 16}, {S("tile"), 1}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 31}, {S("tile"), 1}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 32}, {S("tile"), 1}, {S("block"), 0}});

    applyLayoutAndPrint(layout, {{S("register"), 0}, {S("tile"), 3}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 16}, {S("tile"), 3}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 31}, {S("tile"), 3}, {S("block"), 0}});
    applyLayoutAndPrint(layout, {{S("register"), 32}, {S("tile"), 3}, {S("block"), 0}});

    // applyLayoutAndPrint(layout, {{S("register"), 31}, {S("tile"), 1}, {S("block"), 0}});
    // applyLayoutAndPrint(layout, {{S("register"), 31}, {S("tile"), 0}, {S("block"), 1}});
#endif 
}

}
}

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
