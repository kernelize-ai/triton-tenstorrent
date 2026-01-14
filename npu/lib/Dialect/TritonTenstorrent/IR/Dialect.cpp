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

// TODO: refactor to shared header

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

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

  SmallVector<unsigned> tilesPerCore;
  SmallVector<unsigned> order;
  SmallVector<unsigned> tileShape;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "tilesPerCore") {
      if (parseIntArrayAttr(parser, attr, tilesPerCore,
                            "number of tiles per Tensix core")
              .failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "tileShape") {
      if (parseIntArrayAttr(parser, attr, tileShape, "tile shape").failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  // Assume CTALayout is always default in textual IR -- TODO: consider parsing
  // if we make parseCTAAttr shared
  unsigned rank = order.size();
  auto CTALayout = gpu::CTAEncodingAttr::getDefault(parser.getContext(), rank);

  return parser.getChecked<TiledEncodingAttr>(parser.getContext(), tilesPerCore,
                                              order, tileShape, CTALayout);
}

void TiledEncodingAttr::print(AsmPrinter &printer) const {
  // TODO: print CTALayout?
  printer << "<{"
          << "tilesPerCore = [" << getTilesPerCore() << "]"
          << ", order = [" << getOrder() << "]"
          << ", tileShape = [" << getTileShape() << "]"
          << "}>";
}

SmallVector<unsigned> TiledDotOperandEncodingAttr::getRepOrder() const {
  if (auto parentTiled = dyn_cast<TiledEncodingAttr>(getParent())) {
    return llvm::to_vector(parentTiled.getOrder());
  }
  llvm::report_fatal_error(
      "getRepOrder not implemented for TiledDotOperandEncodingAttr");
  return {};
}

LogicalResult TiledDotOperandEncodingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    unsigned opIdx, Attribute parent) {
  if (opIdx != 0 && opIdx != 1) {
    return emitError()
           << "tttg.tiled_dot_op opIdx parameter can be 0 or 1, got: " << opIdx;
  }
  if (!parent) {
    return emitError() << "tttg.tiled_dot_op parent parameter cannot be null";
  }
  if (!isa<TiledEncodingAttr>(parent)) {
    return emitError()
           << "tttg.tiled_dot_op parent parameter must be a TiledEncodingAttr";
  }
  return success();
}

struct TritonTenstorrentInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  static DialectInferLayoutInterface *
  getTritonGPULayoutInterface(MLIRContext *ctx) {
    auto interface =
        ctx->getOrLoadDialect<gpu::TritonGPUDialect>()
            ->getRegisteredInterface<DialectInferLayoutInterface>();
    assert(interface && "TritonGPU dialect must register a layout interface");
    return interface;
  }

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    return getTritonGPULayoutInterface(getContext())
        ->inferReduceOpEncoding(operandEncoding, axis, resultEncoding, loc);
  }

  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    return getTritonGPULayoutInterface(getContext())
        ->inferTransOpEncoding(operandEncoding, shape, order, resultEncoding,
                               loc);
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> loc) const override {
    return getTritonGPULayoutInterface(getContext())
        ->inferExpandDimsOpEncoding(operandEncoding, axis, resultEncoding, loc);
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> location) const override {
    if (auto dotOpEnc =
            mlir::dyn_cast<TiledDotOperandEncodingAttr>(operandEncoding)) {
      if (opIdx != dotOpEnc.getOpIdx())
        return emitOptionalError(location, "Wrong opIdx");
      if (!isa<TiledEncodingAttr>(retEncoding))
        return emitOptionalError(location,
                                 "Dot with tiled operand encoding's result "
                                 "encoding should be of TiledEncodingAttr");
    } else
      return emitOptionalError(
          location,
          "Dot's a/b's encoding should be of TiledDotOperandEncodingAttr");
    return success();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    auto aEncoding =
        mlir::dyn_cast<TiledDotOperandEncodingAttr>(operandEncodingA);
    auto bEncoding =
        mlir::dyn_cast<TiledDotOperandEncodingAttr>(operandEncodingB);
    if (!aEncoding && !bEncoding)
      return mlir::success();
    if (!aEncoding || !bEncoding)
      return op->emitError("mismatching encoding between A and B operands");

    // Verify that the encodings are valid.
    auto tiledAEncoding =
        dyn_cast_or_null<TiledEncodingAttr>(aEncoding.getParent());
    auto tiledBEncoding =
        dyn_cast_or_null<TiledEncodingAttr>(bEncoding.getParent());
    auto dotOp = cast<DotOp>(op);
    auto resEnc = dotOp.getResult().getType().getEncoding();
    auto tiledResEncoding = dyn_cast<TiledEncodingAttr>(resEnc);

    if (tiledAEncoding || tiledBEncoding || tiledResEncoding) {
      if (!tiledAEncoding || !tiledBEncoding || !tiledResEncoding)
        return op->emitError("mismatching tiled encoding");

      //  auto tiledBEncoding = cast<TiledEncodingAttr>(bEncoding.getParent());
      if (tiledAEncoding.getTileShape() != tiledBEncoding.getTileShape() ||
          tiledAEncoding.getTileShape() != tiledResEncoding.getTileShape()) {
        return op->emitError("mismatched tiled sizes.");
      }
    }
    return success();
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    return getTritonGPULayoutInterface(getContext())
        ->verifyLayoutsAreEqual(shape, expected, got, loc);
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    return getTritonGPULayoutInterface(getContext())
        ->inferReshapeOpEncoding(srcShape, srcEnc, dstShape, dstEnc, loc);
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    return getTritonGPULayoutInterface(getContext())
        ->inferDefaultJoinOpEncoding(srcEnc, dstEnc, shape, loc);
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    return getTritonGPULayoutInterface(getContext())
        ->inferSplitOpEncoding(srcEnc, dstEnc, shape, loc);
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    return getTritonGPULayoutInterface(getContext())
        ->inferFp4ToFpOpEncoding(shape, axis, inEnc, outEnc, fwdInference, loc);
  }
};

void TritonTenstorrentDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "npu/include/Dialect/TritonTenstorrent/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "npu/include/Dialect/TritonTenstorrent/IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonTenstorrentInferLayoutInterface>();
}

} // namespace mlir::triton::npu::tt
