import urllib
import argparse

from PIL import Image

import torch
from torchvision import transforms
# from torchvision.models import resnet50 as resnet
# from torchvision.models import ResNet50_Weights as Weights
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as Weights

from flask import Flask, make_response, send_file, jsonify, request, render_template

app = Flask(__name__)


def log(m): print(m)


def infer_single(img):
    weights = Weights.DEFAULT
    model = resnet(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f'CLASS: {category_name}')


@app.route("/infer/check", methods=['POST'])
def post_em():
    print('infer!!!')
    s = ''
    if request.method == 'POST':
        s += f'<h2>THIS IS A TEST</h2>'
        print(request)
        f = request.files['fileupload']
        img = Image.open(f)
        img = transforms.functional.pil_to_tensor(img)
        print(img.shape)
        infer_single(img)

    return s



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', const=True, default=False, nargs='?', help='debug')
    args = parser.parse_args()
    host, port = '0.0.0.0', 5000
    app.run(host=host, port=port, debug=args.d)
