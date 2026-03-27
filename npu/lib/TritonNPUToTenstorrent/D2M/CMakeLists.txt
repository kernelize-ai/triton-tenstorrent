add_triton_library(TritonNPUToD2M
    TritonNPUToD2M.cpp

    DEPENDS
    TritonNPUToD2MConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
    )
