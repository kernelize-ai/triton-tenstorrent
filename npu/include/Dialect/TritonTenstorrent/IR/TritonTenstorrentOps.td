#ifndef TRITONTENSTORRENT_OPS
#define TRITONTENSTORRENT_OPS

include "TritonTenstorrentDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"

include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure

class TTTS_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonTenstorrent_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTTS_BinaryComputeOp : TTTS_Op<"binary_compute", [Pure, NoMemoryEffect, SameOperandsAndResultShape]> {
  let summary = "Binary compute operation in Tenstorrent compute kernel.";

  let arguments = (ins AnyRankedTensor:$lhs,
                          AnyRankedTensor:$rhs,
                          StrAttr:$opcode);

  let results = (outs Variadic<AnyType>:$result);

  let description = [{
    Coming soon...
  }];

  let assemblyFormat = [{
    `[` $opcode `]` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

#endif
