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
    Wrapper operation for binary compute ops in Tenstorrent compute kernels. `BinaryComputeOp` allows for easy identification of candidate compute ops prior to splitting the Triton kernel into data movement/compute functions and lowering to TTKernel. The existing operation on tensors will be replaced with this generic compute op prior to lowering to TTKernel. In TTKernel, BinaryComputeOp is lowered to the appropriate Tenstorrent tiled compute instruction using the `opcode` attribute.
  }];

  let assemblyFormat = [{
    `[` $opcode `]` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

#endif
