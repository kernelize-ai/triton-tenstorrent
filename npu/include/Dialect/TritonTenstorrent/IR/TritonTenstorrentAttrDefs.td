#ifndef TRITONTENSTORRENT_ATTRDEFS
#define TRITONTENSTORRENT_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"

include "npu/include/Dialect/TritonTenstorrent/IR/TritonTenstorrentDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td" // might not need this?
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td" // might not need this?

def RegisterEncodingAttr : AttrDef<TritonTenstorrent_Dialect, "RegisterEncoding", [DistributedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "register_encoding";

  let description = [{
    A wrapper distributed encoding to hold the desired DEST register index for tiled tensors.

    Instructions using DEST for input/outputs use implicit register indices instead of SSA values to represent register values. This wrapper encoding allows us to track the intended register index for a tensor value.
  }];

  let parameters = (
    ins
    "unsigned":$index,
    "gpu::DistributedEncodingTrait":$parent
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    gpu::CTAEncodingAttr getCTALayout() const {
        return gpu::getCTALayout(getParent());
    }

    SmallVector<unsigned> getRepOrder() const {
        return getParent().getRepOrder();
    }

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const {
        return getParent().toLinearLayout(shape);
    }
  }];
}

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

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const {
        // TODO
        return LinearLayout::empty();
    }
  }];
}

#endif // TRITONTENSTORRENT_ATTRDEFS
