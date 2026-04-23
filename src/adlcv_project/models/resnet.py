from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch



# https://github.com/marcoschouten/hidden-objects/blob/main/distilled_model/model.py
class MultiScaleBackbone(nn.Module):
    """Frozen ResNet-50 backbone with multi-scale feature extraction (C4, C5)."""
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        self.stage5 = resnet.layer4
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        c4 = self.stage4(x)
        c5 = self.stage5(c4)
        return {"c4": c4, "c5": c5}


if __name__ == "__main__":
    model = MultiScaleBackbone()
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    print(f"Output shapes: c4: {out['c4'].shape}, c5: {out['c5'].shape}")