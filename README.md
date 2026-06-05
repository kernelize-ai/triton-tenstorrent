# triton-tenstorrent
Tenstorrent Plugin for Triton

### To install into Triton:

1. Clone this repository (including submodules).
2. Clone the triton repository into a separate directory and checkout the commit hash from `triton.txt`.
3. Initialize a virtualenv (typically inside the Triton repository) and install Triton dependencies.
```
git clone https://github.com/kernelize-ai/triton-tenstorrent.git
cd triton-tenstorrent
git submodule update --init --recursive --depth 0
export TRITON_COMMIT_HASH="$(cat triton.txt)"
cd ..
git clone https://github.com/triton-lang/triton.git
cd triton
git checkout ${TRITON_COMMIT_HASH}

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install -r python/requirements.txt # build-time dependencies

```
4. Install system dependencies for Triton/TritonNPU/Triton Tenstorrent
#### Ubuntu
```
sudo apt-get update && sudo apt-get install -y libboost-fiber-dev
sudo apt install -y build-essential git vim python3 python3-venv python3-dev clang-19 libclang-cpp17-dev libgtest-dev python3-sphinx libnuma-dev numactl libhwloc-dev pkg-config doxygen
```
#### MacOS
```
brew install cmake python@3.12
brew install boost
brew install googletest
brew install act
brew install node
brew install clang-format
```
5. Install Triton.

#### Experimental Tenstorrent TTMLIR Runtime

Triton Tenstorrent supports an experimental lowering path and runtime for compiling Triton kernels to Tenstorrent D2M (Direct to Metal) dialect and then lowering to the Tenstorrent ttnn MLIR dialect which can be interpreted by the TTMLIR runtime. This path is enabled using the flag `TRITON_TTMLIR_TARGET="d2m"`.

To enable the TTMIR runtime integration in Triton we must build Triton against a custom LLVM build with MLIR Python buildings enabled. The script `scripts/build-llvm-with-bindings.sh` will build LLVM with the necessary MLIR Python bindings. Note that you may have to set `TRITON_HOME` if the Triton sources are in a non-standard location. The script also prints out the necessary flags to build Triton with TTMLIR support, which are reproduced below.

Once the custom LLVM version is built, build Triton with the `TTMLIR_RUNTIME_ENABLED=ON` and the `LLVM_BUILD_DIR` flag pointing to the custom LLVM build directory. E.g.:

```
export LLVM_BUILD_DIR="/home/user/.triton/llvm-with-python-bindings/llvm-f6ded0be-ubuntu-x64"
TTMLIR_RUNTIME_ENABLED=ON LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include" LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib" LLVM_SYSPATH=$LLVM_BUILD_DIR  TRITON_VENV_DIR=triton/.venv TRITON_PLUGIN_DIRS=triton-tenstorrent pip install -e . --no-build-isolation
```

Generate a system descriptor with `ttrt`:
```
ttrt query --save-artifacts
```
and note the save path.

Then run the ttnn version of the matmul tutorial:
```
TRITON_TTMLIR_TARGET="d2m" TT_SYSTEM_DESC_PATH="path-to-sys-desc-from-ttrt.ttsys" python triton-tenstorrent/examples/matmul_tensor_descriptor/09-persistent-matmul.py
```

#### Metal Runtime

```
TTMLIR_RUNTIME_ENABLED=ON TRITON_VENV_DIR=/path/to/triton/.venv TRITON_PLUGIN_DIRS=/path/to/triton-tenstorrent pip install -e . --no-build-isolation
```
The flag `TTMLIR_RUNTIME_ENABLED=ON` should only be added to build the `tt-metal` runtime for `tt-mlir` when running on Ubuntu with Tenstorrent silicon.
The flag `TRITON_VENV_DIR` should point to the created virtualenv (in the Triton root directory as shwon above).
`TRITON_PLUGIN_DIRS` tells Triton where to find the Triton Tenstorrent plugin.
6. To generate code for a Tenstorrent device we can use the offline compile script. From the Triton directory:
```
python python/triton/tools/compile.py --target tenstorrent:0:1 --signature "*fp32,*fp32,*fp32,i32,1024" --grid 1,1,1 -n add_kernel ../triton-tenstorrent/python/tutorials/01-vector-add.py
```
The Tenstorrent backend does not yet support a driver (required for running the generated kernels via Triton), so we have copied tutorial 1 and commented out the runtime components. Since the function signature is in untyped python, the types are inferred from the runtime params. The offline compile script is necessary to set the signature for the tutorial manually. The last parameter is the constexpr block size.

The offline compile script will return nothing. The generated code needs to be inspected from the Triton cache directory:

```
ls -nrt ~/.triton/cache
total 4
drwxrwxr-x 2 1000 1000 4096 Nov  6 16:33 VHMLPWTSNPNYU4LIL63Z55TIHAMBIXUNOVPHGSG66CEVUQZTV6NQ
```
Each compiled kernel has its own hashed cache directory, created in order of compilation. Zooming in on the directory from our example:
```
ls -nrt ~/.triton/cache/VHMLPWTSNPNYU4LIL63Z55TIHAMBIXUNOVPHGSG66CEVUQZTV6NQ
total 48
-rw-rw-r-- 1 1000 1000 4972 Nov  6 16:33 add_kernel.source
-rw-rw-r-- 1 1000 1000 3484 Nov  6 16:33 add_kernel.ttir
-rw-rw-r-- 1 1000 1000 3887 Nov  6 16:33 add_kernel.ttgir
-rw-rw-r-- 1 1000 1000 9903 Nov  6 16:33 add_kernel.ttmlir
-rw-rw-r-- 1 1000 1000    4 Nov  6 16:33 add_kernel.so
-rw-rw-r-- 1 1000 1000  580 Nov  6 16:33 add_kernel.json
-rw-rw-r-- 1 1000 1000 5779 Nov  6 16:33 add_kernel.cpp
-rw-rw-r-- 1 1000 1000  847 Nov  6 16:33 __grp__add_kernel.json
```
We find the Triton source file (`ttir`), Triton GPU file (`ttgir`), Triton Tenstorrent MLIR file (`ttmlir`), and generated cpp kernels file (`.cpp`). The `.so` is currently empty.
