add_triton_library(TritonNPUToD2M
    FuncOpToD2MGeneric.cpp
    MemoryOpToD2M.cpp
    TritonNPUToD2M.cpp

    DEPENDS
    TritonNPUToD2MConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
    )
