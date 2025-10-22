add_triton_library(TritonTenstorrentIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonTenstorrentTableGen
  TritonTenstorrentAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
