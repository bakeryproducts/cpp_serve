import torch
import torch_tensorrt
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


def main():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval().to("cuda")
    inp = torch_tensorrt.Input((1, 3, 224, 224))
    trt_model = torch_tensorrt.compile(model, inputs=[inp], enabled_precisions={torch.half})
    torch.jit.save(trt_model, "trt_rn50_224_224.pt")


if __name__ == '__main__':
    main()
