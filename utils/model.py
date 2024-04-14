import torch
import os
from typing import List

def save_model_state_dict(model: torch.nn.Module, path: str) -> None:
    parent = path[0:path.rfind('/')]
    if not os.path.exists(parent):
        os.makedirs(parent)
    
    model_state = model.state_dict()
    torch.save(model_state, path)

def init_conv_weights(model: torch.nn.Module, mean: float=0., std: float=0.02) -> None:
    for _, module in model._modules.items():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            module.weight.data.normal_(mean=mean, std=std)
        elif isinstance(module, torch.nn.Sequential):
            for _, submodule in module._modules.items():
                if isinstance(submodule, torch.nn.Conv2d) or isinstance(submodule, torch.nn.ConvTranspose2d):
                    submodule.weight.data.normal_(mean, std)