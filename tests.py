import time
import requests, json, base64

import fire


def test_python(NUM_BENCH=100, NUM_REQS = 5):
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
        print(f'\t Finished {NUM_REQS} runs: average time per reqest {end / NUM_REQS: 3.3f} ms')
        print('\n\n')


def main(url=1, N=1):
    # url = "http://localhost:7081/predict"
    # bytes64 = base64.b64encode(rawbytes)
    # bytes_string = bytes64.decode('utf-8')
    files = {'fileupload': rawbytes}

    start = time.time()
    for i in range(N):
        # result = requests.post(url, json={"image": bytes_string}).text
        #files = {'media': rawbytes}
        result = requests.post(url, files=files)

    print(result) 
    result = result.content
    result = json.loads(result.decode("utf-8"))
    print(result)
    end = time.time() - start
    print(f'Finished {N} runs: average time {end / N: 3.3f} ms')


if __name__ == '__main__':
    fire.Fire()
