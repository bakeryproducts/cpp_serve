import sys
import json
import time
import base64
import requests

import fire
from tqdm import tqdm
from loguru import logger

handlers = [ {"sink": sys.stdout, "format": "{time:[HH:mm:ss]} - {message}"}, ]
logger.configure(**{"handlers":handlers})



base_url = "http://localhost:7034"
image_path = "data/doggo2.jpg"


def log(url, n_runs, result, avg_time):
    logger.debug(f'\t URL: {url}')
    logger.debug(f'\t Response: {result}')
    logger.info(f'\t Finished {n_runs} runs: average time per reqest {1e3 * avg_time: 3.3f} ms')
    logger.debug("______________________________\n")


def test_all(NUM_BENCH=100, NUM_REQS=5, cpu=False, gpu=False):
    test_cpp_httplib(NUM_BENCH, NUM_REQS, cpu, gpu)
    test_cpp_crow(NUM_BENCH, NUM_REQS, cpu, gpu)
    test_python(NUM_BENCH, NUM_REQS, cpu, gpu)


def test_cpp_httplib(NUM_BENCH=100, NUM_REQS=5, cpu=False, gpu=False):
    assert NUM_BENCH == 100
    assert cpu or gpu
    urls = []
    if cpu:
        urls.append(f'{base_url}/infer_cpp_httplib/cpu/single')
        urls.append(f'{base_url}/infer_cpp_httplib/cpu/bench')
    if gpu:
        urls.append(f'{base_url}/infer_cpp_httplib/gpu/single')
        urls.append(f'{base_url}/infer_cpp_httplib/gpu/bench')

    rawbytes = open(image_path, 'rb').read()
    bytes64 = base64.b64encode(rawbytes)
    bytes_string = bytes64.decode('utf-8')
    files = {'image_file': bytes_string}

    for url in urls:
        start = time.time()
        for i in tqdm(range(NUM_REQS)):
            result = requests.post(url, files=files)
        end = time.time() - start

        result = json.loads(result.text)
        avg_time = end / NUM_REQS
        log(url, NUM_REQS, result, avg_time)


def test_cpp_crow(NUM_BENCH=1, NUM_REQS=1, cpu=False, gpu=False):
    assert NUM_BENCH == 100
    urls = []
    assert cpu or gpu
    if cpu:
        logger.error('i cant figure out dynamic routing in crow, only gpu is available at /infer_cpp_crow/single and /infer_cpp_crow/bench')
        #urls.append(f'{base_url}/infer_cpp_crow/cpu/single')
    if gpu:
        urls.append(f'{base_url}/infer_cpp_crow/single')
        urls.append(f'{base_url}/infer_cpp_crow/bench')

    rawbytes = open(image_path, 'rb').read()
    bytes64 = base64.b64encode(rawbytes)
    bytes_string = bytes64.decode('utf-8')

    for url in urls:
        start = time.time()
        for i in tqdm(range(NUM_REQS)):
            result = requests.post(url, json={"image": bytes_string})
        end = time.time() - start

        result = json.loads(result.text)
        avg_time = end / NUM_REQS
        log(url, NUM_REQS, result, avg_time)


def test_python(NUM_BENCH=100, NUM_REQS=5, cpu=False, gpu=False):
    urls = []
    assert cpu or gpu
    if cpu:
        urls.append(f'{base_url}/infer/cpu/single')
        urls.append(f'{base_url}/infer/cpu/bench_{NUM_BENCH}')
    if gpu:
        urls.append(f'{base_url}/infer/gpu/single')
        urls.append(f'{base_url}/infer/trt/single')
        urls.append(f'{base_url}/infer/gpu/bench_{NUM_BENCH}')
        urls.append(f'{base_url}/infer/trt/bench_{NUM_BENCH}')

    rawbytes = open(image_path, 'rb').read()
    files = {'fileupload': rawbytes}

    for url in urls:
        start = time.time()
        for i in tqdm(range(NUM_REQS)):
            result = requests.post(url, files=files)
        end = time.time() - start

        result = result.content
        result = json.loads(result.decode("utf-8"))
        avg_time = end / NUM_REQS
        log(url, NUM_REQS, result, avg_time)



def main():
    pass


if __name__ == '__main__':
    fire.Fire()
