#ifndef TRITONTENSTORRENT_ATTRDEFS
#define TRITONTENSTORRENT_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"

include "npu/include/Dialect/TritonTenstorrent/IR/TritonTenstorrentDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td" // might not need this?
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td" // might not need this?

def TileEncodingAttr : AttrDef<TritonTenstorrent_Dialect, "TileEncoding"> {
  let mnemonic = "tile_encoding";

  let description = [{
    An encoding to represent the register index of tiles.
  }];

  let parameters = (
    ins
    "unsigned":$index,
    "gpu::DistributedEncodingTrait":$parent
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // TRITONTENSTORRENT_ATTRDEFS
