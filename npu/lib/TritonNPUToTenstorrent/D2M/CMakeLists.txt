add_triton_library(TritonNPUToD2M
    FuncOpToD2MGeneric.cpp
    TritonNPUToD2M.cpp

    DEPENDS
    TritonNPUToD2MConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
    )
