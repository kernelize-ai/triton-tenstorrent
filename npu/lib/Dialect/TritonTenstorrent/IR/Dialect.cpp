#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/TypeID.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"
#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::npu;

#define GET_ATTRDEF_CLASSES
#include "npu/include/Dialect/TritonTenstorrent/IR/AttrDefs.cpp.inc"

namespace mlir::triton::npu::tt {

Attribute RegisterEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};
  unsigned index = cast<IntegerAttr>(attrs.get("index")).getInt();
  auto parent = dyn_cast<gpu::DistributedEncodingTrait>(attrs.get("parent"));
  if (!parent) {
    parser.emitError(parser.getNameLoc(),
                     "expected a distributed encoding trait");
    return {};
  }
  return parser.getChecked<RegisterEncodingAttr>(parser.getContext(), index,
                                                 parent);
}

void RegisterEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "index = " << getIndex() << ", "
          << "parent = " << getParent() << "}>";
}

Attribute TiledEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  // TODO: 
    for (const NamedAttribute &attr : dict) {
#if 1
    if (false) {
#else
    if (attr.getName() == "sizePerThread") {
      if (parseIntArrayAttr(parser, attr, sizePerThread,
                            "number of elements per thread")
              .failed())
        return {};
    } else if (attr.getName() == "threadsPerWarp") {
      if (parseIntArrayAttr(parser, attr, threadsPerWarp,
                            "number of threads per warp")
              .failed())
        return {};
    } else if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA,
                            "number of warps per CTA")
              .failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "CGALayout") {
      ctaAttr = attr.getValue();
#endif
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  return parser.getChecked<TiledEncodingAttr>(parser.getContext(), /*REMOVE*/0u);
}

void TiledEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "ADB TODO" << "}>";
}

gpu::CTAEncodingAttr TiledEncodingAttr::getCTALayout() const {
  MLIRContext *ctx = getContext();
  StringAttr kBlock = StringAttr::get(ctx, "block");

  LinearLayout layout = LinearLayout::empty();
  // TODO: make identity 

  return gpu::CTAEncodingAttr::get(ctx, layout);
}

void TritonTenstorrentDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "npu/include/Dialect/TritonTenstorrent/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "npu/include/Dialect/TritonTenstorrent/IR/Ops.cpp.inc"
      >();
}

} // namespace mlir::triton::npu::tt
