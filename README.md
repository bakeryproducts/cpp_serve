# cpp_serve

web server for torch cpu/gpu cpp/python inferencing/benchmarking

# Usage

```
$ curl --form "fileupload=@data/doggo.jpg" http://localhost:7034/infer/cpu/single
    {"category_name":"malinois","conf":0.7659757733345032}

$ python3 tests.py test_python 20 10

	 URL: http://localhost:7034/infer/cpu/single
	 Response:  {'category_name': 'Rottweiler', 'conf': 0.7541610598564148}
	 Finished 20 runs: average time per reqest  43.079 ms

	 URL: http://localhost:7034/infer/gpu/single
	 Response:  {'category_name': 'Rottweiler', 'conf': 0.754160463809967}
	 Finished 20 runs: average time per reqest  9.330 ms

	 URL: http://localhost:7034/infer/cpu/bench_10
	 Response:  {'average_time_seconds': 0.032080340385437014, 'category_name': 'Rottweiler', 'conf': 0.7541610598564148, 'num_runs': 10}
	 Finished 20 runs: average time per reqest  325.763 ms

	 URL: http://localhost:7034/infer/gpu/bench_10
	 Response:  {'average_time_seconds': 0.0035848140716552733, 'category_name': 'Rottweiler', 'conf': 0.754160463809967, 'num_runs': 10}
	 Finished 20 runs: average time per reqest  48.477 ms



```

# TODO

- ~~cpu/gpu python infer~~
- ~~cpu/gpu cpp infer~~
- ~~force resize in 224 for benchmarking~~
- replace crow, https://github.com/yhirose/cpp-httplib
- integrate cpp into compose
- trt(torchtrt)
- py JIT
- onnx?
- ~~benchmark options for python~~
- benchmark options for cpp
- return messages (same for cpp and python? protobuf?)



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
cpp web server is original [CROW](https://github.com/ipkn/crow) with a lot of fixes, outdated af. Ill look for a better/maintained one. 
cpp torch backend is stable 1.13, this one: libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip

test_api will send N POST json requests with jpg image to the dockerized inferencer. 

Right now its cpu-only JITted resnet18 from torchvision, nothing fancy. 
**Works at speed around 40ms per request**. Now we need to :
1. integrate that into whole compose thing that unifies all inferencing engines
2. do gpu tests
3. replace crow with something

* it looks like crow is leaking memory, but there is an [active fork of crow](https://github.com/CrowCpp/Crow) ...

## 21.01.22

Python-based nearly done. CPU/GPU modes, requests can be :
- single 
- bench_10 for benchmarking **inference** time (10 runs)
- bench_100 ...

tests.py with benchmarking and multiple requests is up
python inference based on torchvision model, without JIT/TRT. thats probably only thing thats not done in python for now.

python processing speed is about **45 ms per request** in single/cpu mode, **9ms/req** in single/gpu

cpp gpu is also up, **350ms/req** on cpu and **10.5ms/req** on gpu. For some reason on devbox machine cpu speed is 6 times slower then on my laptop( ... Still on crow



