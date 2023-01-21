import urllib
import argparse

from PIL import Image

import torch
from torchvision import transforms
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as Weights

from flask import Flask, make_response, send_file, jsonify, request, render_template


def log(m): print(m)


class WrongDeviceError(Exception):
    def __init__(self, device, message="Device is not supported"):
        self.device = device
        self.message = message
        super().__init__(self.message)


class GpuIsNotAvailable(Exception):
    def __init__(self, message="Gpu is not available"):
        self.message = message
        super().__init__(self.message)


app = Flask(__name__)

weights = Weights.DEFAULT
preprocess = weights.transforms()
model_cpu = resnet(weights=weights)
model_cpu.eval()

try:
    model_gpu = model_cpu.cuda()
except Exception as e:
    log(e)
    model_gpu = None


def infer_single(img, model):
    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    log(f'CLASS: {category_name}')
    return category_name, score


def load_to_tensor(f):
    img = Image.open(f)
    img = transforms.functional.pil_to_tensor(img)
    log(img.shape)
    return img


@app.route("/infer/<string:device>/single", methods=['POST'])
def post_em(device):
    res = dict()
    try:
        if device not in ['cpu', 'gpu']:
            raise WrongDeviceError(device)
        if device == 'gpu' and model_gpu is None:
            raise GpuIsNotAvailable

        model = model_cpu if device == 'cpu' else model_gpu
        img = load_to_tensor(request.files['fileupload'])
        class_name, prob = infer_single(img, model)
        res = dict(class_name=class_name, prob=prob)
    except Exception as e:
        res[f'POST_HANDLE_ERROR_{__file__}'] = str(e)

    s = jsonify(res)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', const=True, default=False, nargs='?', help='debug')
    args = parser.parse_args()
    host, port = '0.0.0.0', 5000
    app.run(host=host, port=port, debug=args.d)
