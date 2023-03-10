#!/bin/bash
set -e

rm -rf build
mkdir -p build && cd build
cmake --log-level=VERBOSE -DCMAKE_PREFIX_PATH=libtorch .. 
make -j8 #VERBOSE=1
cd ..
mv build/infer .
rm -rf build
