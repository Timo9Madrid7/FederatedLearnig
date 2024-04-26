import torch
import os
from typing import List, Optional
from torch.utils.data import Dataset, RandomSampler, BatchSampler, DataLoader
from utils.mmd import mmd
from utils.encoding import Encoder

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

def cal_mmd(model: torch.nn.Module, dataset: Dataset, batch_size: int = 128, noise_dim: int = 100, encoder: Optional[Encoder] = None, **mmd_kwargs) -> float:
    device = next(model.parameters()).device.type
    random_sampler = RandomSampler(data_source=dataset, num_samples=batch_size)
    batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)
    target, labels = next(iter(dataloader))

    with torch.no_grad():
        labels = labels.to(device)
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        source: torch.Tensor = model(noise) if not encoder else model(noise, encoder.encode(labels, type='g'))
        source = source.detach().cpu()

    res = mmd(source=source.view(batch_size, -1), target=target.view(batch_size, -1), **mmd_kwargs)
    return res.item()