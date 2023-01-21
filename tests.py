import time
import requests, json, base64

import fire


def test_python(NUM_BENCH=100, NUM_REQS=5):
    urls = [
            'http://localhost:7034/infer/cpu/single',
            'http://localhost:7034/infer/gpu/single',
            f'http://localhost:7034/infer/cpu/bench_{NUM_BENCH}',
            f'http://localhost:7034/infer/gpu/bench_{NUM_BENCH}',
            ]
    image_path = "data/doggo.jpg"
    rawbytes = open(image_path, 'rb').read()
    files = {'fileupload': rawbytes}

    for url in urls:
        start = time.time()
        for i in range(NUM_REQS):
            result = requests.post(url, files=files)
        end = time.time() - start

        result = result.content
        result = json.loads(result.decode("utf-8"))
        print(f'\t URL: {url}')
        print('\t Response: ', result)
        print(f'\t Finished {NUM_REQS} runs: average time per reqest {1e3 * end / NUM_REQS: 3.3f} ms')
        print()


def main():
    pass


if __name__ == '__main__':
    fire.Fire()
