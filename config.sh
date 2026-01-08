#!/bin/bash

set -e

LLVM_VERSION="20.1.8"
LLVM_PROJECT_DIR="externalDeps/llvm-project-${LLVM_VERSION}.src"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check for compilers
if command -v clang &> /dev/null && command -v clang++ &> /dev/null; then
  C_COMPILER=clang
  CXX_COMPILER=clang++
  echo "Using clang/clang++ as compilers"
else
  C_COMPILER="${CC:-cc}"
  CXX_COMPILER="${CXX:-c++}"
  echo "clang not found, using default compilers: $C_COMPILER/$CXX_COMPILER"
fi

# Check for build system
if command -v ninja &> /dev/null; then
  BUILD_SYSTEM="Ninja"
  BUILD_COMMAND="ninja"
  echo "Using Ninja build system"
else
  BUILD_SYSTEM="Unix Makefiles"
  BUILD_COMMAND="make"
  echo "Ninja not found, using Make build system"
fi

# Check for linker
if command -v ld.lld &> /dev/null || command -v lld &> /dev/null; then
  LINKER_OPTION="-DLLVM_USE_LINKER=lld"
  echo "Using lld as linker"
else
  LINKER_OPTION=""
  echo "lld not found, using default linker"
fi

download() {
  echo "Downloading LLVM ${LLVM_VERSION}..."
  "${SCRIPT_DIR}/scripts/download_llvm.sh"
  echo "LLVM downloaded and extracted."

}

genDefs() {
  if [ ! -d "build" ]; then
    echo "No build folder found! Run 'config' first."
    exit 1
  fi
  echo "Generating RISC-V definitions..."
  ./build/bin/llvm-tblgen -dump-json \
    -I "${LLVM_PROJECT_DIR}/llvm/include" \
    -I "${LLVM_PROJECT_DIR}/llvm/lib/Target/RISCV" \
    "${LLVM_PROJECT_DIR}/llvm/lib/Target/RISCV/RISCV.td" \
    > riscv_defs.json
  echo "Instructions dumped to riscv_defs.json"
}

conf() {
  # Check if LLVM project exists
  if [ ! -d "$LLVM_PROJECT_DIR" ]; then
    echo "LLVM project not found. Downloading..."
    download
  fi

  echo "Configuring LLVM build..."
  CC="$C_COMPILER" CXX="$CXX_COMPILER" cmake \
    -S "${LLVM_PROJECT_DIR}/llvm" \
    -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DLLVM_TARGETS_TO_BUILD='RISCV' \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_ENABLE_PROJECTS='' \
    ${LINKER_OPTION} \
    -G"${BUILD_SYSTEM}"
  cp build/compile_commands.json .
  echo "Configuration complete."
}

build() {
  if [ ! -d "build" ]; then
    echo "No build folder found! Run 'config' first."
    exit 1
  fi
  cd build
  $BUILD_COMMAND llvm-mc llvm-tblgen
  cd ..
}

build_all() {
  if [ ! -d "build" ]; then
    echo "No build folder found! Run 'config' first."
    exit 1
  fi
  cd build
  $BUILD_COMMAND
  cd ..
}

clean() {
  echo "Cleaning build artifacts..."
  rm -rf build
  rm -f compile_commands.json
  echo "Clean complete."
}

mrproper() {
  echo "Performing mrproper (deep clean)..."
  clean
  echo "Removing contents of externalDeps..."
  if [ -d "externalDeps" ]; then
    find externalDeps -mindepth 1 -maxdepth 1 -not -name '.gitignore' -exec rm -rf {} +
  fi
  echo "MrProper complete."
}

case $1 in
download | d)
  download
  ;;
genDefs | gendefs | gd)
  genDefs
  ;;

config | c)
  conf
  ;;
build | b)
  build
  ;;
buildall | build-all | ba)
  build_all
  ;;
clean)
  clean
  ;;
mrproper | distclean)
  mrproper
  ;;
*)
  if [ $1 ]; then
    echo "Unknown argument: $1"
  fi
  echo "Script to configure and build:"
  echo "  d | download               Download LLVM ${LLVM_VERSION} source."
  echo "  gd| genDefs                Generate RISC-V definitions JSON."
  echo "  c | config                 Configure for Development (auto-downloads if needed)."
  echo "  b | build                  Build the llta target."
  echo "  ba| build-all              Build all targets."
  echo "  clean                      Remove build artifacts."
  echo "  mrproper | distclean       Deep clean (removes build + externalDeps)."
  exit
  ;;
esac
