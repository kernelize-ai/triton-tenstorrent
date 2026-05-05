add_subdirectory(D2M)

add_triton_library(TritonNPUToTenstorrent
    ComputeOpToTTKernel.cpp
    DotOpToTTKernel.cpp
    DropFunctionPass.cpp
    ElementwiseOpsToTTKernel.cpp
    IntrinsicOpToTTKernel.cpp
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

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
