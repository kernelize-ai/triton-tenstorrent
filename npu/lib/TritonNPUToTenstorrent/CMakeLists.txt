add_triton_library(TritonNPUToTenstorrent
    ComputeOpToTTKernel.cpp
    DotOpToTTKernel.cpp
    DropFunctionPass.cpp
    ElementwiseOpsToTTKernel.cpp
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
