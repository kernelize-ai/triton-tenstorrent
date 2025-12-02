add_triton_library(TritonNPUToTenstorrent
    ComputeOpToTTKernel.cpp
    DropFunctionPass.cpp
    ElementwiseOpsToTTKernel.cpp
    FinalizeCBTransaction.cpp
    MakeRangeOpToTTKernel.cpp
    MemoryOpToTTKernel.cpp
    PointerInfoAnalysis.cpp
    SPMDOpToTTKernel.cpp
    TritonFuncOpToFuncOp.cpp
    TritonNPUToTTKernel.cpp
    Utility.cpp
    ViewOpToTTKernel.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
