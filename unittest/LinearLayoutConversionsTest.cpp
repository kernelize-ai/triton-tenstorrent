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
    return tt::TiledEncodingAttr::get(&ctx, tilesPerCore, order, tileShape);
  }

  tt::TiledDotOperandEncodingAttr dot(tt::TiledEncodingAttr parent,
                                      unsigned idx) {
    return tt::TiledDotOperandEncodingAttr::get(&ctx, idx, parent);
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

void applyLayoutAndPrint(LinearLayout &layout,
                         ArrayRef<std::pair<StringAttr, int32_t>> ins) {
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
  auto layout = toLinearLayout({32, 32}, enc);

  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"),
                             {{0, 1},
                              {0, 2},
                              {0, 4},
                              {0, 8},
                              {0, 16},
                              {1, 0},
                              {2, 0},
                              {4, 0},
                              {8, 0},
                              {16, 0}}},
                            {S("tile"), {}},
                            {S("block"), {}},
                        },
                        {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, Tiled_MultiTile) {
  auto enc = tiled({2, 2}, {32, 32}, {1, 0});
  auto layout = toLinearLayout({64, 64}, enc);

  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"),
                             {{0, 1},
                              {0, 2},
                              {0, 4},
                              {0, 8},
                              {0, 16},
                              {1, 0},
                              {2, 0},
                              {4, 0},
                              {8, 0},
                              {16, 0}}},
                            {S("tile"), {{0, 32}, {32, 0}}},
                            {S("block"), {}},
                        },
                        {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, Tiled_Dot_Lhs) {
  auto parent = tiled({8, 2}, {32, 32}, {1, 0});
  auto dotOperand = dot(parent, /*idx=*/0);
  auto layout = toLinearLayout({256, 64}, dotOperand);

  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"),
                             {{0, 1},
                              {0, 2},
                              {0, 4},
                              {0, 8},
                              {0, 16},
                              {1, 0},
                              {2, 0},
                              {4, 0},
                              {8, 0},
                              {16, 0}}},
                            {S("tile"), {{0, 32}, {32, 0}, {64, 0}, {128, 0}}},
                            {S("block"), {}},
                        },
                        {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, Tiled_Dot_Rhs) {
  auto parent = tiled({2, 4}, {32, 32}, {1, 0});
  auto dotOperand = dot(parent, /*idx=*/1);
  auto layout = toLinearLayout({64, 128}, dotOperand);

  EXPECT_EQ(layout, LinearLayout(
                        {
                            {S("register"),
                             {{0, 1},
                              {0, 2},
                              {0, 4},
                              {0, 8},
                              {0, 16},
                              {1, 0},
                              {2, 0},
                              {4, 0},
                              {8, 0},
                              {16, 0}}},
                            {S("tile"), {{32, 0}, {0, 32}, {0, 64}}},
                            {S("block"), {}},
                        },
                        {S("dim0"), S("dim1")}));
}

} // namespace
} // namespace mlir::triton::npu

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
