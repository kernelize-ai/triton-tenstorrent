add_triton_ut(
  NAME TenstorrentLinearLayoutConversions
  SRCS LinearLayoutConversionsTest.cpp
  LIBS
    TritonTenstorrentIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
