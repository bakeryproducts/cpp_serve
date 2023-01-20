rm -rf build

mkdir -p build
cd build
cmake --log-level=VERBOSE -DCMAKE_PREFIX_PATH=libtorch .. 
make -j4 #VERBOSE=1
cd ..

mv build/infer .
rm -rf build
