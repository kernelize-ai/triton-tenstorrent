#pragma once

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/StringRef.h"

#define SPMD_ARGS(X) X(x_grid) X(y_grid)

namespace mlir::triton::npu {

enum class SpmdArg {
#define X(n) n,
  SPMD_ARGS(X)
#undef X
      Count
};

inline llvm::StringRef spmdArgName(SpmdArg a) {
  static const char *names[] = {
#define X(n) #n,
      SPMD_ARGS(X)
#undef X
  };
  return names[(int)a];
}
// trailing convention: the SPMD args are the last `Count` arguments
inline mlir::Value getSpmdArg(mlir::FunctionOpInterface f, SpmdArg a) {
  unsigned base = f.getNumArguments() - (unsigned)SpmdArg::Count;
  return f.getArgument(base + (unsigned)a);
}

} // namespace mlir::triton::npu
