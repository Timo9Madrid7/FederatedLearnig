from typing import List, Dict, Optional, Union
from torch.utils.data import DataLoader

import torch


class Node:

    def __init__(self, model_structure: List) -> None:
        self.model_structure = model_structure
        self.weights = None
    
    def get_weights(self) -> List[float]:
        return self.weights
    
    def set_weights(self, weights: List[float]) -> None:
        self.weights = weights.copy()

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
    

class GenNode:

    def __init__(self, model_structure: Dict[str, List[int]]) -> None:
        self.model_structure: Dict[str, List[int]] = model_structure
        self.weights: Dict[str, List[float]] = {'d': [], 'g': []}
        self.gradients: Dict[str, List[float]] = {'d': [0 for _ in range(self.model_structure[-1])], 'g': [0 for _ in range(self.model_structure[-1])]}

    def get_weights(self, type: Optional[str]=None) -> Union[Dict[str, List[float]], List[float]]:
        if type is None:
            return self.weights
        return self.weights[type]
    
    def set_weights(self, weights: Union[Dict[str, List[float]], List[float]], type: Optional[str]=None) -> None:
        if type is None:
            self.weights = weights.copy()
        else:
            self.weights[type] = weights.copy()

    def get_gradients(self, type: Optional[str]=None) -> Union[Dict[str, List[float]], List[float]]:
        if type is None:
            return self.gradients
        return self.gradients[type]
    
    def set_gradients(self, gradients: Union[Dict[str, List[float]], List[float]], type: Optional[str]=None) -> None:
        if type is None:
            self.gradients = gradients.copy()
        else:
            self.gradients[type] = gradients.copy()