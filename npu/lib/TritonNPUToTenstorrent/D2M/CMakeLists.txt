add_triton_library(TritonNPUToD2M
    ArgConversionHelper.cpp
    ComputeOpToD2M.cpp
    DotOpToD2M.cpp
    FuncOpToD2MGeneric.cpp
    FuncOpToTTNNGeneric.cpp
    MemoryOpToD2M.cpp
    ReduceOpToD2M.cpp
    SPMDOpToD2M.cpp
    TensorizeScalars.cpp
    TritonNPUToD2M.cpp

    DEPENDS
    TritonNPUToD2MConversionPassIncGen

    LINK_LIBS
    ${TTMLIR_LIBS}
    TritonNPUAnalysis
    )
