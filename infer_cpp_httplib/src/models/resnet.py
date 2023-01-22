from PIL import Image
import numpy as np
from functools import partial

import torch
import torchvision
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as Weights




class InferWrap(torch.nn.Module):
    def __init__(self, use_original_preprocess=False):
        super().__init__()
        self.weights = Weights.DEFAULT
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        #self.model = resnet(weights=self.weights)
        self.model.eval()

        if use_original_preprocess:
            self.preprocess = lambda x: self.weights.transforms()(x).unsqueeze(0)
        else:
            self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(3,1,1), requires_grad=False)
            self.std  = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(3,1,1), requires_grad=False)
            self.preprocess = self._preprocess_on_stats

    def _preprocess_on_stats(self, x):
        x = x / 255.
        x = (x - self.mean) / self.std
        x = x.unsqueeze(0)
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bilinear')
        return x

    def forward_logits(self, img):
        batch = self.preprocess(img)
        pred = self.model(batch)
        return pred

    @torch.no_grad()
    def forward(self, img):
        pred = self.forward_logits(img).squeeze(0)
        pred = pred.softmax(0)
        class_id = torch.tensor(pred.argmax())
        score = torch.tensor(pred[class_id])
        # category_name = self.weights.meta["categories"][class_id]
        return torch.stack([class_id, score])


def export_model(model, xb, device):
    model = model.to(device)
    xb = xb.to(device)
    traced_script_module = torch.jit.trace(model, xb)
    traced_script_module.save(f"model_{device}.pth")


def main():
    model = InferWrap()
    xb = (torch.ones(3, 224, 224)*255).int()
    export_model(model, xb, 'cpu')
    print('TEST START')
    test(model)
    if torch.cuda.is_available:
        export_model(model, xb, 'cuda')


def test(model):
    f = '/data/doggo1.jpg'
    img = np.array(Image.open(f))
    img = torch.from_numpy(img)
    batch = img.permute(2,0,1)

    category_name, score = model(batch)
    print(f'CLASS: {category_name}, {score}')


if __name__ == '__main__':
    main()
