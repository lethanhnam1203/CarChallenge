import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_s,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
    ViT_B_32_Weights,
    vit_b_32,
)
from torch import nn


def get_small_nn(
    input_dim: int,
    immediate_dim: int,
    output_dim: int = 1,
    dropout_prob: float = 0.33,
) -> nn.Module:
    """Create a small neural network with two fully connected layers and a dropout layer in between"""
    return nn.Sequential(
        nn.Linear(input_dim, immediate_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_prob),
        nn.Linear(immediate_dim, output_dim),
        nn.Sigmoid(),
    )


class ResNetWithDualFC(nn.Module):
    def __init__(self, freeze_backbone: bool = True, immediate_dim: int = 256) -> None:
        super(ResNetWithDualFC, self).__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        num_feats = backbone.fc.in_features
        # Define two separate fully connected layers for each block
        self.fc_hood = get_small_nn(num_feats, immediate_dim)
        self.fc_backdoor = get_small_nn(num_feats, immediate_dim)
        # Replace the backbone's fully connected layer with a dummy module
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        out_hood = self.fc_hood(x)
        out_backdoor = self.fc_backdoor(x)
        stacked_outputs = torch.cat((out_hood, out_backdoor), dim=1)
        return stacked_outputs


class EfficientNetWithDualFC(nn.Module):
    def __init__(self, freeze_backbone: bool = True, immediate_dim: int = 128) -> None:
        super(EfficientNetWithDualFC, self).__init__()
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        num_feats = backbone.classifier[-1].in_features
        self.fc_hood = get_small_nn(num_feats, immediate_dim=immediate_dim)
        self.fc_backdoor = get_small_nn(num_feats, immediate_dim=immediate_dim)
        backbone.classifier[-1] = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        out_hood = self.fc_hood(x)
        out_backdoor = self.fc_backdoor(x)
        stacked_outputs = torch.cat((out_hood, out_backdoor), dim=1)
        return stacked_outputs


class MobileNetWithDualFC(nn.Module):
    def __init__(self, freeze_backbone: bool = True, immediate_dim: int = 128) -> None:
        super(MobileNetWithDualFC, self).__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        num_feats = backbone.classifier[-1].in_features
        self.fc_hood = get_small_nn(num_feats, immediate_dim=immediate_dim)
        self.fc_backdoor = get_small_nn(num_feats, immediate_dim=immediate_dim)
        backbone.classifier[-1] = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        out_hood = self.fc_hood(x)
        out_backdoor = self.fc_backdoor(x)
        stacked_outputs = torch.cat((out_hood, out_backdoor), dim=1)
        return stacked_outputs


class ViTWithDualFC(nn.Module):
    def __init__(self, freeze_backbone: bool = True, immediate_dim: int = 128) -> None:
        super(ViTWithDualFC, self).__init__()
        backbone = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        num_feats = backbone.heads.head.in_features
        self.fc_hood = get_small_nn(num_feats, immediate_dim=immediate_dim)
        self.fc_backdoor = get_small_nn(num_feats, immediate_dim=immediate_dim)
        backbone.heads.head = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        out_hood = self.fc_hood(x)
        out_backdoor = self.fc_backdoor(x)
        stacked_outputs = torch.cat((out_hood, out_backdoor), dim=1)
        return stacked_outputs
