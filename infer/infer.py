import time
import copy
from PIL import Image
from functools import partial

import torch
from torchvision import transforms
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as Weights

from errors import GpuIsNotAvailable, WrongDeviceError, WrongModeError, BenchMarkTotalError, MAX_BENCH_NUM, MIN_BENCH_NUM


def _log(m): print(m)


weights = Weights.DEFAULT
preprocess = weights.transforms()
model_cpu = resnet(weights=weights)
model_cpu.eval()

try:
    model_gpu = copy.deepcopy(model_cpu)
    model_gpu.cuda()
except Exception as e:
    model_gpu = None


def _preprocess_on_stats(x, mean, std):
    x = x / 255.
    x = (x - mean) / std
    x = x.unsqueeze(0)
    x = torch.nn.functional.interpolate(x, (224, 224), mode='bilinear')
    return x


def infer_single(img, model):
    device = next(model.parameters()).device
    img = img.to(device)
    orig_prepr = False
    if orig_prepr:
        batch = preprocess(img).unsqueeze(0)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
        batch = _preprocess_on_stats(img, mean, std)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    res = dict(category_name=category_name, conf=score)
    return res


def load_to_tensor(f):
    img = Image.open(f)
    img = transforms.functional.pil_to_tensor(img)
    _log(img.shape)
    return img


def get_model(device):
    if device not in ['cpu', 'gpu']:
        raise WrongDeviceError(device)
    if device == 'gpu' and model_gpu is None:
        raise GpuIsNotAvailable
    model = model_cpu if device == 'cpu' else model_gpu
    return model


def infer_benchmark(*args, N, base, **kwargs):
    start = time.time()
    r = dict()
    for _ in range(N):
        r = base(*args, **kwargs)
    average_time = (time.time() - start) / N
    _log(f'Running benchmark: {N} runs {average_time:3.5f} s each')

    assert isinstance(r, dict)
    r['average_time_seconds'] = average_time
    r['num_runs'] = N
    return r


def get_infer(mode):
    """
        mode : 'single' or 'bench_<int>'
    """
    if mode != 'single' and not mode.startswith('bench_'):
        raise WrongModeError(mode)
    if mode == 'single':
        # single mode, run 1 infer on image
        infer = infer_single
    else:
        # benchmark mode, run N infers on same image and time it
        N = int(mode.split('_')[1])
        if N <= MIN_BENCH_NUM or N > MAX_BENCH_NUM:
            raise BenchMarkTotalError(N)
        infer = partial(infer_benchmark, N=N, base=infer_single)

    return infer
