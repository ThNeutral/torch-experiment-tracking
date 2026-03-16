import torchvision
import torch
from torch import nn
from utils import set_seeds

def create_effnetb3(
    out_features: int,
    device: torch.device | str, 
    seed: int = 42
):
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    model = torchvision.models.efficientnet_b3(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    set_seeds(seed)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=out_features)
    ).to(device)

    model.name = "effnetb3"
    print(f"[INFO] Created new {model.name} model.")
    return model, weights