import torch
import torch.nn as nn
import torch.nn.functional as F
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

    # hook
    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor, grad_cam = False) -> torch.Tensor:
        x = self.features(x)
        if grad_cam:
            x.register_hook(self.activation_hook)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def grad_cam(self, x, real_class = None, rectify = True, return_logits = False):
        # x: (batch_size, 3, 224, 224) # real_class: (batch_size)
        distribution = self.forward(x, grad_cam=True)
        
        pred = distribution.argmax(dim=1)

        if real_class is None:
            distribution[:, pred].backward(torch.ones_like(distribution[:, pred]))
        else:
            # pred: (batch_size)
            assert pred.shape == real_class.shape, "pred and real_class must have the same shape"
            distribution[:, real_class].backward(torch.ones_like(distribution[:, real_class]))
            
        gradients = self.get_activation_gradient()
        # gradients: (batch_size, last_conv_layer_channels, 7, 7)

        # pool the gradients across the channel
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # pooled_gradients: (last_conv_layer_channels)

        # activations of the last conv layer
        activations = self.get_activation(x).detach()
        # activations: (batch_size, last_conv_layer_channels, 7, 7)

        # weight the channels by corresponding gradients
        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average all channels of the weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        # heatmap: (batch_size, 7, 7)
        
        # Verificar que no hay nan
        assert torch.isnan(heatmap).sum() == 0, "Hay nan en el heatmap"

        # relu to obtain only positive effect
        if rectify:
            heatmap = F.relu(heatmap)
            # normalize the heatmap between 0 and 1 with min-max normalization
            min_b, _ = torch.min(heatmap.view(heatmap.shape[0], -1), dim=1) # min_b: (batch_size)
            max_b, _ = torch.max(heatmap.view(heatmap.shape[0], -1), dim=1) # max_b: (batch_size)
            assert min_b.shape == (x.shape[0],), f"min_b must have the same shape as batch_size, {min_b.shape} != {x.shape[0]}"
            assert max_b.shape == (x.shape[0],), f"max_b must have the same shape as batch_size, {max_b.shape} != {x.shape[0]}"
            div = max_b - min_b
            div[div == 0] = 1
            heatmap = (heatmap - min_b.view(-1, 1, 1)) / div.view(-1, 1, 1)
            
        else:
            # Normalizar por el máximo en valor absoluto
            max_abs, _ = torch.max(torch.abs(heatmap.view(heatmap.shape[0], -1)), dim=1) # max_abs: (batch_size)
            max_abs[max_abs == 0] = 1
            assert max_abs.shape == (x.shape[0],), f"max_abs must have the same shape as batch_size, {max_abs.shape} != {x.shape[0]}"
            # Copiar el max_abs para que tenga la misma forma que el heatmap (7, 7)
            heatmap /= max_abs.view(-1, 1, 1)
        
        if return_logits:
            return heatmap, pred, distribution
        
        return heatmap, pred
        
    
    # extract gradient
    def get_activation_gradient(self):
        return self.gradients

    # extract the activation after the last ReLU
    def get_activation(self, x):
        return self.features(x)
    
    def set_blocks(self):
        current_block = []
        self.block_counter = 0
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                block_name = f"block{self.block_counter}"
                setattr(self, block_name, nn.Sequential(*current_block))
                self.block_counter += 1
                current_block = []
            else:
                current_block.append(layer)
        block_name = f"block{self.block_counter}"
        setattr(self, block_name, self.classifier)
        self.block_counter += 1
        return self.block_counter
            
    def __len__(self):
        return self.block_counter
    

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
    


def _vgg(batch_norm: bool, pretrained: bool, progress: bool, mini: bool, path: str, mini_clf: bool) -> VGG:
    if mini:
        model = VGG(make_layers([64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], batch_norm=batch_norm), mini_clf=mini_clf)
        return model
    model = VGG(make_layers([64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], batch_norm=batch_norm))
    if pretrained:
        if path == '':
            # To use this, you have to download weights from https://download.pytorch.org/models/vgg19_bn-c79401a0.pth and store them into checkpoints directory
            model.load_state_dict(torch.load('../checkpoints/Vgg19_Weights_bn-c79401a0/vgg19_bn-c79401a0.pth'))  
        else:
            model.load_state_dict(torch.load(path))
    return model

def vgg19_bn(pretrained: bool = False, progress: bool = True, mini: bool = False, path: str = '', mini_clf: bool = False) -> VGG:
    """VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(True, pretrained, progress, mini, path, mini_clf)