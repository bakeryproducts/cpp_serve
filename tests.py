import time
import requests, json, base64

import fire


base_url = "http://localhost:7034"
image_path = "data/doggo2.jpg"


def log(url, n_runs, result, avg_time):
    print(f'\t URL: {url}')
    print('\t Response: ', result)
    print(f'\t Finished {n_runs} runs: average time per reqest {1e3 * avg_time: 3.3f} ms')
    print()



def test_cpp_crow(NUM_BENCH=1, NUM_REQS=1):
    urls = [
        f'{base_url}/infer_cpp/single',
    ]
    rawbytes = open(image_path, 'rb').read()
    bytes64 = base64.b64encode(rawbytes)
    bytes_string = bytes64.decode('utf-8')

    for url in urls:
        start = time.time()
        for i in range(NUM_REQS):
            result = requests.post(url, json={"image": bytes_string})
        end = time.time() - start

        result = json.loads(result.text)
        avg_time = end / NUM_REQS
        log(url, NUM_REQS, result, avg_time)


def test_python(NUM_BENCH=100, NUM_REQS=5):
    urls = [
        f'{base_url}/infer/cpu/single',
        f'{base_url}/infer/gpu/single',
        f'{base_url}/infer/trt/single',
        f'{base_url}/infer/cpu/bench_{NUM_BENCH}',
        f'{base_url}/infer/gpu/bench_{NUM_BENCH}',
        f'{base_url}/infer/trt/bench_{NUM_BENCH}',
    ]

    rawbytes = open(image_path, 'rb').read()
    files = {'fileupload': rawbytes}

    for url in urls:
        start = time.time()
        for i in range(NUM_REQS):
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
