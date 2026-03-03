import torch
import torch.nn as nn
from torchvision import models, transforms

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load pre-trained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # We only need the first few layers to capture fur texture (edges/colors)
        # Layer 35 is usually the sweet spot for "style" and "content"
        self.slice = nn.Sequential(*list(vgg.children())[:36]).eval()
        for param in self.parameters():
            param.requires_grad = False
        
        self.l1 = nn.L1Loss()
        self.imagenet_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.to(device)

    def forward(self, pred, target):
        # Normalize to imagenet norms
        pred = torch.stack([self.imagenet_norm(p) for p in pred], dim=0)
        target = torch.stack([self.imagenet_norm(t) for t in target], dim=0)

        pred_features = self.slice(pred)
        target_features = self.slice(target)
        
        return self.l1(pred_features, target_features)