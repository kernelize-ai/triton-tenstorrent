add_triton_library(TritonTenstorrentIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Ops.cpp

  DEPENDS
  TritonTenstorrentTableGen
  TritonTenstorrentAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
