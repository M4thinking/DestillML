# Description:
# - This file contains the model definition for VGG based on the original paper by Simonyan and Zisserman.
# - Also, it contains the definition of the ResNet model from the original paper by He et al.

# 🍦 Vanilla PyTorch
import torch
from torch import nn

# 📦 Other Libraries
from typing import Any

class ResNet(nn.Module):
    pass

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, mini_clf = False) -> None:
        super().__init__()
        self.mini_clf = mini_clf
        self.features = features
        self.block_counter = 0
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if self.mini_clf == False:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 2048),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(2048, num_classes),
            )
            
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                    
    def device(self):
        return next(self.parameters()).device
    
    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor, grad_cam = False) -> torch.Tensor:
        # FOR GRAD-CAM is Necessary: has features, register_hook, and activation_hook
        x = self.features(x) # 1. features
        if grad_cam:
            x.register_hook(self.activation_hook) # 2. register_hook -> activation_hook
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# Crear el modelo
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)

def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


if __name__ == "__main__":
    ckpt_path = "checkpoints/vgg19/vgg19_bn.pth"
    teacher = vgg19_bn(pretrained=False, progress=True, num_classes=1000, init_weights=True, dropout=0.5)
    teacher.load_state_dict(torch.load(ckpt_path))
    params = sum(p.numel() for p in teacher.parameters()) # 143.678.248 (143M)
    assert params==143678248, f"El modelo tiene {params} parametros"
