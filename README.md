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

All urls:

**python**
- http://localhost:\<PORT\>/infer/cpu/single
- http://localhost:\<PORT\>/infer/cpu/bench_\<INT\>
- http://localhost:\<PORT\>/infer/gpu/single
- http://localhost:\<PORT\>/infer/gpu/bench_\<INT\>
- http://localhost:\<PORT\>/infer/trt/single
- http://localhost:\<PORT\>/infer/trt/bench_\<INT\>

**cpp crow, gpu ony**
- http://localhost:\<PORT\>/infer_cpp_crow/single
- http://localhost:\<PORT\>/infer_cpp_crow/bench

**cpp cpp-httplib**
- http://localhost:\<PORT\>/infer_cpp_httplib/cpu/single
- http://localhost:\<PORT\>/infer_cpp_httplib/cpu/bench
- http://localhost:\<PORT\>/infer_cpp_httplib/gpu/single
- http://localhost:\<PORT\>/infer_cpp_httplib/gpu/bench





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
```

### Tests
```
# pip3 install -r requirements.txt

# test all modes
python3 tests.py test_all 

# test all gpu modes with 100 benchmark passes and 50 requests each
python3 tests.py test_all 100 50 --gpu

# test python only, gpu modes with 1 benchmark run and 1 request
python3 tests.py test_python 1 1 --gpu

# see tests.py for more

```

```
$ python3 tests.py test_all 100 50 --gpu
01-23 13:17:15 | DEBUG    | 	 URL: http://localhost:7034/infer_cpp_httplib/gpu/single
requests: 100%|█████████████████████████████████| 50/50 [00:00<00:00, 63.73it/s]
01-23 13:17:16 | DEBUG    | 	 Response: {'Prediction': 'Rottweiler', 'Confidence': '0.968502'}
01-23 13:17:16 | INFO     | 	 Finished 50 runs: average time per reqest  15.737 ms
01-23 13:17:16 | DEBUG    | ______________________________

01-23 13:17:16 | DEBUG    | 	 URL: http://localhost:7034/infer_cpp_httplib/gpu/bench
requests: 100%|█████████████████████████████████| 50/50 [00:38<00:00,  1.30it/s]
01-23 13:17:54 | DEBUG    | 	 Response: {'Prediction': 'Rottweiler', 'Confidence': '0.968502', 'average_time_seconds': '0.007263', 'num_runs': '100'}
01-23 13:17:54 | INFO     | 	 Finished 50 runs: average time per reqest  770.095 ms
01-23 13:17:54 | DEBUG    | ______________________________

01-23 13:17:54 | DEBUG    | 	 URL: http://localhost:7034/infer_cpp_crow/single
requests: 100%|█████████████████████████████████| 50/50 [00:00<00:00, 62.40it/s]
01-23 13:17:55 | DEBUG    | 	 Response: {'Status': 'Success', 'Confidence': '0.968502', 'Prediction': 'Rottweiler'}
01-23 13:17:55 | INFO     | 	 Finished 50 runs: average time per reqest  16.034 ms
01-23 13:17:55 | DEBUG    | ______________________________

01-23 13:17:55 | DEBUG    | 	 URL: http://localhost:7034/infer_cpp_crow/bench
requests: 100%|█████████████████████████████████| 50/50 [00:58<00:00,  1.16s/it]
01-23 13:18:53 | DEBUG    | 	 Response: {'average_time_seconds': '0.011014', 'Status': 'Success', 'Confidence': '0.968502', 'Prediction': 'Rottweiler'}
01-23 13:18:53 | INFO     | 	 Finished 50 runs: average time per reqest  1162.222 ms
01-23 13:18:53 | DEBUG    | ______________________________

01-23 13:18:53 | DEBUG    | 	 URL: http://localhost:7034/infer/gpu/single
requests: 100%|█████████████████████████████████| 50/50 [00:00<00:00, 96.24it/s]
01-23 13:18:54 | DEBUG    | 	 Response: {'category_name': 'Rottweiler', 'conf': 0.9685019254684448}
01-23 13:18:54 | INFO     | 	 Finished 50 runs: average time per reqest  10.399 ms
01-23 13:18:54 | DEBUG    | ______________________________

01-23 13:18:54 | DEBUG    | 	 URL: http://localhost:7034/infer/trt/single
requests: 100%|████████████████████████████████| 50/50 [00:00<00:00, 135.08it/s]
01-23 13:18:54 | DEBUG    | 	 Response: {'category_name': 'Rottweiler', 'conf': 0.9683527946472168}
01-23 13:18:54 | INFO     | 	 Finished 50 runs: average time per reqest  7.412 ms
01-23 13:18:54 | DEBUG    | ______________________________

01-23 13:18:54 | DEBUG    | 	 URL: http://localhost:7034/infer/gpu/bench_100
requests: 100%|█████████████████████████████████| 50/50 [00:23<00:00,  2.13it/s]
01-23 13:19:18 | DEBUG    | 	 Response: {'average_time_seconds': 0.0045995450019836424, 'category_name': 'Rottweiler', 'conf': 0.9685019254684448, 'num_runs': 100}
01-23 13:19:18 | INFO     | 	 Finished 50 runs: average time per reqest  468.894 ms
01-23 13:19:18 | DEBUG    | ______________________________

01-23 13:19:18 | DEBUG    | 	 URL: http://localhost:7034/infer/trt/bench_100
requests: 100%|█████████████████████████████████| 50/50 [00:11<00:00,  4.40it/s]
01-23 13:19:29 | DEBUG    | 	 Response: {'average_time_seconds': 0.0031868886947631837, 'category_name': 'Rottweiler', 'conf': 0.9683527946472168, 'num_runs': 100}
01-23 13:19:29 | INFO     | 	 Finished 50 runs: average time per reqest  227.394 ms
01-23 13:19:29 | DEBUG    | ______________________________

```

## curl Requests

### Python
```
$ curl --form "fileupload=@data/doggo2.jpg" http://localhost:7034/infer/cpu/single
    {"category_name":"Rottweiler","conf":0.9685025215148926}

$ curl --form "fileupload=@data/doggo2.jpg" http://localhost:7034/infer/gpu/bench_321
    {"average_time_seconds":0.004574902703828901,"category_name":"Rottweiler","conf":0.9685019254684448,"num_runs":321}

$ curl --form "fileupload=@data/doggo2.jpg" http://localhost:7034/infer/trt/bench_1000
    {"average_time_seconds":0.001876108169555664,"category_name":"Rottweiler","conf":0.9683527946472168,"num_runs":1000}

```

### cpp-httplib

```
$ curl --form "image_file=@data/doggo2.jpg" http://localhost:7034/infer_cpp_httplib/gpu/single
    {"Prediction":"Rottweiler","Confidence":"0.968503"}

$ curl --form "image_file=@data/doggo2.jpg" http://localhost:7034/infer_cpp_httplib/gpu/single
    {"Prediction":"Rottweiler","Confidence":"0.968502"}

$ curl --form "image_file=@data/doggo2.jpg" http://localhost:7034/infer_cpp_httplib/cpu/bench
    {"Prediction":"Rottweiler","Confidence":"0.968503","average_time_seconds":"0.066293","num_runs":"100"}

$ curl --form "image_file=@data/doggo2.jpg" http://localhost:7034/infer_cpp_httplib/gpu/bench
    {"Prediction":"Rottweiler","Confidence":"0.968502","average_time_seconds":"0.007234","num_runs":"100"}
```

### cpp-crow

cpp-crow runs on base64 images in json, not sure how to do that from curl...


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
- ~~some docs~~
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

## 23.01.22

Docs, cleanup. There are quite a lot of things to do:
 - Less bloated docker images
 - cpp infer without base64 encoding
 - ...
