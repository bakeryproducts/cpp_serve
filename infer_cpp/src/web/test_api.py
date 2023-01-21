import time
import requests, json, base64

url = "http://localhost:7081/predict"
image_path = "../../data/doggo2.jpg"
rawbytes = open(image_path, 'rb').read()
bytes64 = base64.b64encode(rawbytes)
bytes_string = bytes64.decode('utf-8')

start = time.time()
N = 10
for i in range(N):
    result = requests.post(url, json={"image": bytes_string}).text
total = time.time() - start
print(json.loads(result))
avg = 1e3 * total / N
print(f'Finished {N} runs: average time {avg: 5.3f} ms')
