# cpp_serve

web server for torch cpu/gpu cpp/python inferencing/benchmarking

- python-based inference, 6 modes: cpu/gpu/trt single/benchmark
- cpp-based inference, [crow](https://github.com/ipkn/crow) as webserver, 2 modes: cpu/gpu single 
- cpp-based inference, [cpp-httplib](https://github.com/yhirose/cpp-httplib) as webserver, 4 modes: cpu/gpu single / benchmark

Default model in tests is Resnet50:
```
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
```

### Modes

Single mode is single ingerence with one reqest and one image:
```
$curl --form "fileupload=@data/doggo1.jpg" http://localhost:7034/infer/cpu/single
    {"category_name":"Rottweiler","conf":0.9683527946472168}
```

Benchmark mode is for testing actual inference speed, additionally measuring forward propagation time.
```
$curl --form "fileupload=@data/doggo1.jpg" http://localhost:7034/infer/cpu/bench_100
    {"average_time_seconds":0.0021880578994750975,"category_name":"Rottweiler","conf":0.9683527946472168,"num_runs":100}
```

CPU mode is for cpu inference

GPU mode is for gpu inference

TRT mode is for gpu inference with tensorrt-compiled model


# Usage

## Start

### prebuilt TRT engine (GPU)

This will:
 * pull NGC pytorch docker , pull RN50 model from hub
 * compile trt engine into TS torch model, fixed img size 224x224
 * save .pt model inplace
```
cd build_trt && ./start.sh
```

### Main framework

Docker compose with various types of inference for testing latency
```
make build
make start
python3 tests.py test_all
```

## Requests

```
$ curl --form "fileupload=@infer_cpp/data/doggo2.jpg" http://localhost:7034/infer/trt/bench_1000
    {"average_time_seconds":0.001876108169555664,"category_name":"Rottweiler","conf":0.9683527946472168,"num_runs":1000}

$ python3 tests.py test_python 100 20

	 URL: http://localhost:7034/infer/cpu/single
	 Response:  {'category_name': 'Rottweiler', 'conf': 0.9685025215148926}
	 Finished 20 runs: average time per reqest  91.552 ms

	 URL: http://localhost:7034/infer/gpu/single
	 Response:  {'category_name': 'Rottweiler', 'conf': 0.9685019254684448}
	 Finished 20 runs: average time per reqest  20.077 ms

	 URL: http://localhost:7034/infer/trt/single
	 Response:  {'category_name': 'Rottweiler', 'conf': 0.9683527946472168}
	 Finished 20 runs: average time per reqest  6.978 ms

	 URL: http://localhost:7034/infer/cpu/bench_100
	 Response:  {'average_time_seconds': 0.07384127140045166, 'category_name': 'Rottweiler', 'conf': 0.9685025215148926, 'num_runs': 100}
	 Finished 20 runs: average time per reqest  7653.253 ms

	 URL: http://localhost:7034/infer/gpu/bench_100
	 Response:  {'average_time_seconds': 0.004479460716247559, 'category_name': 'Rottweiler', 'conf': 0.9685019254684448, 'num_runs': 100}
	 Finished 20 runs: average time per reqest  462.572 ms

	 URL: http://localhost:7034/infer/trt/bench_100
	 Response:  {'average_time_seconds': 0.001861283779144287, 'category_name': 'Rottweiler', 'conf': 0.9683527946472168, 'num_runs': 100}
	 Finished 20 runs: average time per reqest  192.562 ms


```

# TODO

- ~~cpu/gpu python infer~~
- ~~cpu/gpu cpp infer~~
- ~~force resize in 224 for benchmarking~~
- ~~replace crow, https://github.com/yhirose/cpp-httplib~~
- ~~integrate cpp into compose~~
- ~~trt(torchtrt) for python~~    
- py JIT
- onnx?
- ~~benchmark options for python~~
- ~~benchmark options for cpp~~
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

## 22.01.22

TRT for python-based inf, moved to [cpp-httplib](https://github.com/yhirose/cpp-httplib) as cpp webserver, 30%-ish percent more requests on gpu then crow-based. 
Also there is single/bench cpu/gpu modes for cpp-httplib infer. 
And there is a test_all option for testing all modes

