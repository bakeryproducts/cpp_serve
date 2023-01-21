import time
import requests, json, base64

import fire


def main(url=1, N=1):
    # url = "http://localhost:7081/predict"
    url = 'http://localhost:7034/infer/cpu/single'
    image_path = "data/doggo.jpg"
    rawbytes = open(image_path, 'rb').read()
    # bytes64 = base64.b64encode(rawbytes)
    # bytes_string = bytes64.decode('utf-8')

    start = time.time()
    for i in range(N):
        # result = requests.post(url, json={"image": bytes_string}).text
        files = {'media': rawbytes}
        result = requests.post(url, files=files)

    print(json.loads(result))
    end = time.time() - start
    print(f'Finished {N} runs: average time {end / N: 3.3f} ms')


if __name__ == '__main__':
    fire.Fire(main)
