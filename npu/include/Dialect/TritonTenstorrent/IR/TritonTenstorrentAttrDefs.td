#ifndef TRITONTENSTORRENT_ATTRDEFS
#define TRITONTENSTORRENT_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"

include "npu/include/Dialect/TritonTenstorrent/IR/TritonTenstorrentDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td" // might not need this?
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td" // might not need this?

def TileEncodingAttr : AttrDef<TritonTenstorrent_Dialect, "TileEncoding", [DistributedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "tile_encoding";

  let description = [{
    A wrapper distributed encoding to hold the desired tile register index for tiled tensors in L1/SRAM.
  }];

  let parameters = (
    ins
    "unsigned":$index,
    "gpu::DistributedEncodingTrait":$parent
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    gpu::CTALayoutAttr getCTALayout() const {
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

#endif // TRITONTENSTORRENT_ATTRDEFS
