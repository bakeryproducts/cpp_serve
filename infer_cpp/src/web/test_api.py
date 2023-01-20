import requests, json, base64

url = "http://localhost:7081/predict"
image_path = "../../data/doggo2.jpeg"
rawbytes = open(image_path, 'rb').read()
bytes64 = base64.b64encode(rawbytes)
bytes_string = bytes64.decode('utf-8')

import time
start = time.time()
N = 20
for i in range(N):
    result = requests.post(url, json={"image": bytes_string}).text
    print(json.loads(result))
end = time.time() - start
print(f'Finished {N} runs: average time {end / N: 3.3f} ms')
