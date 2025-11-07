#ifndef TRITONTENSTORRENT_DIALECT
#define TRITONTENSTORRENT_DIALECT

include "mlir/IR/OpBase.td"

def TritonTenstorrent_Dialect : Dialect {
  let name = "triton_tenstorrent";

  let cppNamespace = "::mlir::triton::npu::tt";

  let description = [{
    Triton Tenstorrent Dialect. Helpers ops for progressive lowering from TritonGPU IR to Tenstorrent TTKernel IR.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
  ];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
