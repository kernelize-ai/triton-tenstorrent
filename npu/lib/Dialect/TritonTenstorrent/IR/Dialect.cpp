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

Attribute TileEncodingAttr::parse(AsmParser &parser, Type type) {
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
  return parser.getChecked<TileEncodingAttr>(parser.getContext(), index,
                                             parent);
}

void TileEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "index = " << getIndex() << ", "
          << "parent = " << getParent() << "}>";
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
