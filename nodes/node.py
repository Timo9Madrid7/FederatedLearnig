from typing import List
from torch.utils.data import DataLoader

import torch


class Node:

    def __init__(self, model_structure: List) -> None:
        self.model_structure = model_structure
        self.weights = None
    
    def get_weights(self) -> List[float]:
        return self.weights
    
    def set_weights(self, weights: List[float]) -> None:
        self.weights = weights

    def evaluate_accuracy(self, data_iter: DataLoader, model: torch.nn.Module, device: str=None) -> float:
        weights = self.get_weights()

        for layer, param in enumerate(model.parameters(), 1):
            param.data = torch.tensor(weights[self.model_structure[layer-1]:self.model_structure[layer]]).view(param.data.size())

        model = model.to(device)

        acc_sum, n = 0.0, 0

        with torch.no_grad():
            
            model.eval()
            
            for X, y in data_iter:                    
                acc_sum += (model(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
                n += y.shape[0]

            model.train()

        return acc_sum / n