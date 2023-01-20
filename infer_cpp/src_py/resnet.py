import cv2
import numpy as np
from functools import partial

import torch
import torchvision
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as Weights


def _preprocess_on_stats(x, mean, std):
    x = x/255.
    x = (x - mean) / std
    x = x.unsqueeze(0)
    return x


class InferWrap(torch.nn.Module):
    def __init__(self, use_original_preprocess=False):
        super().__init__()
        self.weights = Weights.DEFAULT
        self.model = resnet(weights=self.weights)
        self.model.eval()

        if use_original_preprocess:
            self.preprocess = lambda x: self.weights.transforms()(x).unsqueeze(0)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            self.preprocess = partial(_preprocess_on_stats, mean=mean, std=std)

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


def test(model):
    f = '/data/doggo1.jpg'
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    batch = img.permute(2,0,1)

    category_name, score = model(batch)
    print(f'CLASS: {category_name}, {score}')

if __name__ == '__main__':
    main()
