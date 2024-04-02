import torch
import os

def save_model_state_dict(model: torch.nn.Module, path: str) -> None:
    parent = path[0:path.rfind('/')]
    if not os.path.exists(parent):
        os.makedirs(parent)
    
    model_state = model.state_dict()
    torch.save(model_state, path)