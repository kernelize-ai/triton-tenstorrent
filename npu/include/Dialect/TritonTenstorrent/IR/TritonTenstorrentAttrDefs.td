#ifndef TRITONTENSTORRENT_ATTRDEFS
#define TRITONTENSTORRENT_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "npu/include/Dialect/TritonTenstorrent/IR/TritonTenstorrentDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td" // might not need this?
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td" // might not need this?

/// Attribute Enumerations
def ReaderFunc : I32EnumCase<"ReaderFunc", 0, "reader_func">;
def WriterFunc : I32EnumCase<"WriterFunc", 1, "writer_func">;
def ComputeFunc : I32EnumCase<"ComputeFunc", 2, "compute_func">;

def DerivedFuncType : I32Enum<"DerivedFuncType",
                                    "Tenstorrent derived function type",
                                    [ReaderFunc, WriterFunc, ComputeFunc]> {
  let cppNamespace = "::mlir::triton::npu::tt";
}

def DerivedFuncTypeAttr : EnumAttr<TritonTenstorrent_Dialect, DerivedFuncType,
                                         "derived_func_type"> {
  let assemblyFormat = "$value";
}

/// Memory encodings
def TiledEncodingAttr : AttrDef<TritonTenstorrent_Dialect, "TiledEncoding", [DistributedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "tiled_encoding";

  let description = [{
    A distributed encoding to represent tiled tensors in L1/SRAM.
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$tilesPerCore,
    ArrayRefParameter<"unsigned">:$order,
    ArrayRefParameter<"unsigned">:$tileShape,

    "gpu::CTAEncodingAttr":$CTALayout
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<unsigned>":$tilesPerCore,
                     "ArrayRef<unsigned>":$order), [{
      // Default tile shape to (32, 32) if not provided
      SmallVector<unsigned, 2> tileShape(2, 32);

      return TiledEncodingAttr::get(context, tilesPerCore, order, tileShape);
    }]>,

    AttrBuilder<(ins "ArrayRef<unsigned>":$tilesPerCore,
                     "ArrayRef<unsigned>":$order,
                     "ArrayRef<unsigned>":$tileShapeIn), [{
      // Default tile shape to (32, 32) if empty tile shape provided
      SmallVector<unsigned, 2> tileShape(tileShapeIn.begin(), tileShapeIn.end());
      if (tileShapeIn.size() == 0)
        tileShape = SmallVector<unsigned, 2>(2, 32);

      auto rank = order.size();

      SmallVector<unsigned, 2> ctasPerCGA(rank, 1);
      SmallVector<unsigned, 2> ctaSplitNum(rank, 1);
      gpu::CTAEncodingAttr CTALayout = gpu::CTAEncodingAttr::fromSplitParams(context, ctasPerCGA, ctaSplitNum, order);
      return $_get(context, tilesPerCore, order, tileShape, CTALayout);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    SmallVector<unsigned> getRepOrder() const {
        return SmallVector<unsigned>(getOrder());
    }

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  }];
}

def TiledDotOperandEncodingAttr : AttrDef<TritonTenstorrent_Dialect, "TiledDotOperandEncoding", [DistributedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "tiled_dot_op";

  let description = [{
    Idential to TritonGPU::DotOperandEncodingAttr but for Tiled parent attrs.
  }];

  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent
  );

  let assemblyFormat = "`<` `{` struct(params) `}` `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    SmallVector<unsigned> getRepOrder() const;

    gpu::CTAEncodingAttr getCTALayout() const {
        return cast<npu::tt::TiledEncodingAttr>(getParent()).getCTALayout();
    }

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  }];
}

#endif // TRITONTENSTORRENT_ATTRDEFS
