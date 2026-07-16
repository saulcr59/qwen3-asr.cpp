#!/bin/bash
set -e

CMAKE_X86="/usr/local/bin/cmake"

if [ ! -f "$CMAKE_X86" ]; then
  echo "ERROR: x86_64 cmake not found at $CMAKE_X86"
  echo 'Install with: arch -x86_64 /usr/local/bin/brew install cmake'
  exit 1
fi

# Clean
rm -rf build-arm build-intel ggml/build
mkdir -p build-arm build-intel

########################################
echo "--- 1/5: Build ggml (ARM64) ---"
########################################
cd ggml
rm -rf build
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64

make -j$(sysctl -n hw.logicalcpu)
cd ../..

########################################
echo "--- 2/5: Build app (ARM64) ---"
########################################
cd build-arm

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64

make -j$(sysctl -n hw.logicalcpu)

mv qwen3-asr-cli qwen3-asr-cli-arm64
cd ..

########################################
echo "--- 3/5: Rebuild ggml (x86_64 via Rosetta) ---"
########################################
rm -rf ggml/build

cd ggml
mkdir build && cd build

arch -x86_64 $CMAKE_X86 .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=x86_64

arch -x86_64 make -j$(sysctl -n hw.logicalcpu)

cd ../..

########################################
echo "--- 4/5: Build app (x86_64 via Rosetta) ---"
########################################
cd build-intel

arch -x86_64 $CMAKE_X86 .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=x86_64

arch -x86_64 make -j$(sysctl -n hw.logicalcpu)

mv qwen3-asr-cli qwen3-asr-cli-x86_64
cd ..

########################################
echo "--- 5/5: Create universal binary ---"
########################################
lipo -create \
    build-arm/qwen3-asr-cli-arm64 \
    build-intel/qwen3-asr-cli-x86_64 \
    -output qwen3-asr-cli

echo "--- DONE ---"
lipo -info qwen3-asr-cli
