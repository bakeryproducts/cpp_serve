# cpp_serve

web server for torch cpu/gpu cpp/python inferencing/benchmarking

# Change log

## 20.01.22

Things are kinda working rn, infer_cpp docker is on: 

```
cd infer_cpp
docker build -t infer_cpp .
docker run -p 7081:8181 infer_cpp /src/test.sh
docker run -p 7081:8181 infer_cpp /src/web/start.sh
cd infer_cpp/src/web/ && python3 test_api.py
```
cpp web server is original CROW with a lot of fixes, outdated af. Ill look for a better/maintained one. 
cpp torch backend is stable 1.13, this one: libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip

test_api will send N POST json requests with jpg image to the dockerized inferencer. 

Right now its cpu-only JITted resnet18 from torchvision, nothing fancy. 
*Works at speed around 5ms per request*. Now we need to :
1. integrate that into whole compose thing that unifies all inferencing engines
2. do gpu tests
3. replace crow with something
