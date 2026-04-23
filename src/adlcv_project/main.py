from torch import nn
import torch


from adlcv_project.models.resnet import MultiScaleBackbone



if __name__ == "__main__":
    model = MultiScaleBackbone()
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    print(f"Output shapes: c4: {out['c4'].shape}, c5: {out['c5'].shape}")