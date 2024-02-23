# Description:
# - This module contains the definition of the XAI class that is used to perform Grad-CAM, Shapley, and other possible XAI methods.

# 🍦 Vanilla PyTorch
import torch
from torch.nn import functional as F
from torch import nn

# 👀 Torchvision for CV
from torchvision import transforms

# 🎯 Shap Values
import shap

# 📚 Other libraries
import numpy as np
class XAI(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor, grad_cam = False) -> torch.Tensor:
        return self.model(x, grad_cam=grad_cam)
    
    def get_activation_gradient(self):
        return self.model.gradients
    
    def get_activation(self, x):
        return self.model.features(x)

    def grad_cam(self, x, real_class = None, rectify = True, return_logits = False):
        # x: (batch_size, 3, 224, 224) # real_class: (batch_size)
        distribution = self.forward(x, grad_cam=True)
        
        pred = distribution.argmax(dim=1)

        if real_class is None:
            distribution[:, pred].backward(torch.ones_like(distribution[:, pred]))
        else:
            assert pred.shape == real_class.shape, "pred and real_class must have the same shape" # pred: (batch_size)
            distribution[:, real_class].backward(torch.ones_like(distribution[:, real_class]))
        
        # get the gradients of the activations of the last conv layer
        gradients = self.get_activation_gradient() # (batch_size, last_conv_layer_channels, last_conv_layer_W, last_conv_layer_H)

        # pool the gradients across the channel
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # (batch_size, last_conv_layer_channels) (alpha_k)

        # activations of the last conv layer
        activations = self.get_activation(x).detach() # (batch_size, last_conv_layer_channels, last_conv_layer_W, last_conv_layer_H) (A_k)

        # weight the channels by corresponding gradients
        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i] # (L_k = A_k * alpha_k)

        # average all channels of the weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze() # (batch_size, last_conv_layer_W, last_conv_layer_H) (sum_k(L_k)/K)
        
        # Verificar que no hay nan
        assert torch.isnan(heatmap).sum() == 0, "Hay nan en el heatmap"

        # relu to obtain only positive effect and normalize between 0 and 1
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
            # Copiar el max_abs para que tenga la misma forma que el heatmap (batch_size, W, H)
            heatmap /= max_abs.view(-1, 1, 1)
        
        if return_logits:
            return heatmap, pred, distribution
        
        return heatmap, pred
    
    def shap(self, X, to_explain, class_names, inv_normalize = None, true_labels = None):
        # X: (N1, 3, 224, 224) # to_explain: (N2, 3, 224, 224)
        # explainer = shap.DeepExplainer(self.model, X)
        # shap_values, indexes = explainer.shap_values(to_explain, ranked_outputs=2)
        
        explainer = shap.GradientExplainer((self.model, self.model.layer4), X)
        shap_values, indexes = explainer.shap_values(to_explain, ranked_outputs=2, nsamples=500)
        
        index_names = np.vectorize(lambda x: class_names[x])(indexes)
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
        print(len(shap_values), shap_values[0].shape)
        # plot the feature attributions in numpy image format (H, W, C)
        to_explain = inv_normalize(to_explain).numpy()
        to_explain = np.swapaxes(np.swapaxes(to_explain, 1, -1), 1, 2)
        shap.image_plot(shap_values, to_explain, index_names, True, true_labels)
    
    
if __name__ == "__main__":
    pass
    # # 🧠 Models
    # from VGG import vgg19_bn
    # from PIL import Image
    
    # ckpt_path = "checkpoints/vgg19/vgg19_bn.pth"
    # teacher = vgg19_bn(pretrained=False, progress=True, num_classes=1000, init_weights=True, dropout=0.5)
    # teacher.load_state_dict(torch.load(ckpt_path))
    # teacher.eval()
    
    # # {0: 'tench, Tinca tinca', 1: 'goldfish, Carassius auratus', ...}
    # with open("./data/imagenet_classes.txt") as f:
    #     labels = [line.strip().split(":")[-1].strip() for line in f.readlines()]
    
    # img_path1 = "data/dog1.png"
    # img_path2 = "data/dog2.png"
    # true_labels = ['basenji', 'golden retriever']
    # img1 = Image.open(img_path1).convert("RGB")
    # img2 = Image.open(img_path2).convert("RGB")

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    
    # inv_normalize = transforms.Normalize(
    #     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    #     std=[1/0.229, 1/0.224, 1/0.225]
    # )
    
    # imgs = torch.stack([transform(img1), transform(img2)])
    # preds = teacher(imgs)
    # explainer = XAI(teacher)
    # heatmaps, preds = explainer.grad_cam(imgs)
        
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import cv2 as cv
    
    # imgs = inv_normalize(imgs)
    
    # fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    # for i in range(2):
    #     img = imgs[i].permute(1, 2, 0).detach().numpy()
    #     heatmap = heatmaps[i].detach().numpy()
    #     heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
    #     heatmap = np.uint8(255 * heatmap)
    #     heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    #     superimposed_img = heatmap * 0.5 + img * 255
    #     superimposed_img = superimposed_img / np.max(superimposed_img)
    #     ax[0, i].imshow(img)
    #     ax[1, i].imshow(superimposed_img)
    #     ax[0, i].axis("off")
    #     ax[1, i].axis("off")
    #     ax[0, i].set_title(f"True label: {true_labels[i]}")
    #     ax[1, i].set_title(f"Predicted label: {labels[preds[i]]}")
         
    # plt.tight_layout()
    # plt.show()

    # # SHAP
    # import json
    # url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    # fname = shap.datasets.cache(url)
    # with open(fname) as f:
    #     class_names = json.load(f)
    
    # from datasets import ImageNet
    # dataset = ImageNet(root="./data/imagenet/", split="train", transform=transform)
    # X, _ = dataset[0:50]
    # to_explain, y = dataset[60:62]
    # true_labels = [class_names[str(y[i].item())][1] for i in range(y.shape[0])]
    # explainer.shap(X, to_explain, class_names, inv_normalize, true_labels)
    