#ifndef TRITONTENSTORRENT_OPS
#define TRITONTENSTORRENT_OPS

include "TritonTenstorrentDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike
include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure

def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTTS_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonTenstorrent_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTTS_BinaryComputeOp : TTTS_Op<"binary_compute", [Pure, SameOperandsAndResultShape, SameOperandsAndResultEncoding]> {
  let summary = "Binary compute operation in Tenstorrent compute kernel.";

  let arguments = (ins AnyRankedTensor:$lhs,
                          AnyRankedTensor:$rhs,
                          StrAttr:$opcode);

  let results = (outs Variadic<AnyType>:$result);

  let description = [{
    Wrapper operation for binary compute ops in Tenstorrent compute kernels. `BinaryComputeOp` allows for easy
    identification of candidate compute ops prior to splitting the Triton kernel into data movement/compute functions
    and lowering to TTKernel. The existing operation on tensors will be replaced with this generic compute op prior
    to lowering to TTKernel. In TTKernel, BinaryComputeOp is lowered to the appropriate Tenstorrent tiled compute
    instruction using the `opcode` attribute.
  }];

  let assemblyFormat = [{
    `[` $opcode `]` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def TTTS_MulticastOp : TTTS_Op<"multicast", [SingleBlock]> {
  let summary = "Evaluate region on root and multicast the result to scope";

  let description = [{
    TODO
  }];

  let arguments = (ins Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$dest);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $dest $body attr-dict `:` functional-type(operands, results)
  }];

  // TODO: verifier
}

def TTTS_YieldOp : TTTS_Op<"yield", [HasParent<"MulticastOp">, Pure, Terminator, ReturnLike]> {
  let summary = "yield from the default region of `ttts.multicast`";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^)? attr-dict (`:` type($values)^)?";
}

#endif
