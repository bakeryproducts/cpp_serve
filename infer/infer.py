import time
import copy
from PIL import Image
from functools import partial

import torch
from torchvision import transforms
# from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as Weights

from errors import GpuIsNotAvailable, WrongDeviceError, WrongModeError, BenchMarkTotalError, MAX_BENCH_NUM, MIN_BENCH_NUM, TRTIsNotAvailable


def _log(m): print(m)


weights = Weights.DEFAULT
preprocess = weights.transforms()
model_cpu = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model_cpu = resnet(weights=weights) # it should be this one, but accuracy on torchvision model are lower
model_cpu.eval()
model_cpu.get_device = lambda:'cpu'

try:
    model_gpu = copy.deepcopy(model_cpu)
    model_gpu.cuda()
    model_gpu.get_device = lambda:'cuda'
except Exception as e:
    model_gpu = None

try:
    import torch_tensorrt # linked to trt engine
    model_trt = torch.jit.load("/mnt/trt/trt_rn50_224_224.pt")
    model_trt.get_device = lambda:'cuda'
except Exception as e:
    print(f"CANNOT BUILD TRT MODEL, {e}")
    model_trt = None


def _preprocess_on_stats(x, mean, std):
    """ tensor preprocess for nn, instead of torchvision
       preprocess, torchvision is broken on JIT right now

    Args:
        x (torch.tensor): image
        mean (torch.tensor): [3,1,1] imagenet mean, const
        std (torch.tensor): [3,1,1] imagenet std, const

    Returns:
        torch.tensor: preprocessed image
    """
    x = x / 255.
    x = (x - mean) / std
    x = x.unsqueeze(0)
    x = torch.nn.functional.interpolate(x, (224, 224), mode='bilinear')
    return x


@torch.no_grad()
def infer_single(img, model):
    """ Inference for single image

    Args:
        img (torch.tensor): raw image tensor, [0-255] int8
        model (torch.nn.Module): torch model, gpu/cpu/trt-compiled engine

    Returns:
        dict: postprocessed prediction (softmax, argmax, etc) from model {'category_name': cat, 'conf'= .65}
    """
    device = model.get_device()
    img = img.to(device)
    orig_prepr = False
    if orig_prepr:
        batch = preprocess(img).unsqueeze(0)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
        batch = _preprocess_on_stats(img, mean, std)

    #with torch.autocast(device_type=device):
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
    if device == 'cpu':
        model = model_cpu
    elif device == 'gpu':
        if model_gpu is None: raise GpuIsNotAvailable
        model = model_gpu
    elif device == 'trt':
        if model_trt is None: raise TRTIsNotAvailable
        model = model_trt
    else:
        raise WrongDeviceError(device)

    return model


def infer_benchmark(*args, N, base, **kwargs):
    """ Inference with timed forward pass, N times

    Args:
        N (int): how many times do a forward pass to time it
        base (function): function to time

    Returns:
        dict: return value from base function, extended with stats from multiple run
        that was timed: 'average_time_seconds', 'num_runs'
    """
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
    """ Select inference mode

    Args:
        mode (str): string mode, can be 'single' or 'bench_<int>'

    Raises:
        WrongModeError: unknown mode
        BenchMarkTotalError: bench_<int> int N should be in range of [MIN, MAX]

    Returns:
        function: inference function (infer_benchmark or infer_single)
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
