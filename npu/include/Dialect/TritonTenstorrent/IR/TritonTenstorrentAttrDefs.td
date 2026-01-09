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

// TODO: terrible name conflicts ahead
def TiledEncodingAttr : AttrDef<TritonTenstorrent_Dialect, "TiledEncoding", [DistributedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "tiled_encoding";

  let description = [{
    A distributed encoding to represent tiled tensors in L1/SRAM.
  }];

  let parameters = (
    ins
    "unsigned":$tileSize
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    SmallVector<unsigned> getRepOrder() const {
        // TODO 
        return {}; 
    }

    gpu::CTAEncodingAttr getCTALayout() const;

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const {
        // TODO
        return LinearLayout::empty();
    }
  }];
}

#endif // TRITONTENSTORRENT_ATTRDEFS
