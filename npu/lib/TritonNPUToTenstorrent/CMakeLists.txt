add_triton_library(TritonNPUToTenstorrent
    ComputeOpToTTKernel.cpp
    DropFunctionPass.cpp
    ElementwiseOpsToTTKernel.cpp
    FinalizeCBTransaction.cpp
    MemoryOpToTTKernel.cpp
    SPMDOpToTTKernel.cpp
    TritonFuncOpToFuncOp.cpp
    TritonNPUToTTKernel.cpp
    Utility.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
