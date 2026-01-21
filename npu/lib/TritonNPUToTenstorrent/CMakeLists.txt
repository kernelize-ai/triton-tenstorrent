add_triton_library(TritonNPUToTenstorrent
    ComputeOpToTTKernel.cpp
    DotOpToTTKernel.cpp
    DropFunctionPass.cpp
    ElementwiseOpsToTTKernel.cpp
    MakeRangeOpToTTKernel.cpp
    MemoryOpToTTKernel.cpp
    PointerInfoAnalysis.cpp
    PrintOpToTTKernel.cpp
    SPMDOpToTTKernel.cpp
    TritonFuncOpToFuncOp.cpp
    TritonNPUToTTKernel.cpp
    TypeConverter.cpp
    Utility.cpp
    ViewOpToTTKernel.cpp

    CreateTTNNGenericOp.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
