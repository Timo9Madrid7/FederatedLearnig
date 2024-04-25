from typing import List, Tuple, Dict
from nodes.node import Node, GenNode
import torch

class Server(Node):

    def __init__(self, model_structure: List) -> None:
        super().__init__(model_structure)

    def naive_aggregation(self, weight_list: List[List[float]]) -> List[float]:
        weight_list : torch.Tensor = torch.tensor(weight_list, dtype=torch.float32, requires_grad=False)
        self.weights = weight_list.mean(0).data.tolist()
        return self.weights
    

class GenServer(GenNode):

    def __init__(self, model_structure: torch.Dict[str, List[int]]) -> None:
        super().__init__(model_structure)
        self._m_: Dict[str, torch.Tensor] = {'d': torch.zeros((model_structure['d'][-1], ), dtype=torch.float32, requires_grad=False), 'g': torch.zeros((model_structure['g'][-1], ), dtype=torch.float32, requires_grad=False)}
        self._v_: Dict[str, torch.Tensor] = {'d': torch.zeros((model_structure['d'][-1], ), dtype=torch.float32, requires_grad=False), 'g': torch.zeros((model_structure['g'][-1], ), dtype=torch.float32, requires_grad=False)}
        self._t_: Dict[str, int] = {'d': 0, 'g': 0}

    def naive_aggregation(self, weight_list: List[List[float]], type: str) -> List[float]:
        weight_list : torch.Tensor = torch.tensor(weight_list, dtype=torch.float32, requires_grad=False)
        self.weights[type] = weight_list.mean(0).data.tolist()
        return self.weights[type]
    
    def weighted_aggregation(self, weight_list: List[List[float]], coef_list: List[float], type: str, softmax: bool=False) -> List[float]:
        weight_list : torch.Tensor = torch.tensor(weight_list, dtype=torch.float32, requires_grad=False)
        coef_list: torch.Tensor =  torch.tensor(coef_list, dtype=torch.float32, requires_grad=False)
        if softmax:
            coef_list = torch.softmax(coef_list, dim=0)
        self.weights[type] = weight_list.T.matmul(coef_list).data.tolist()
        return self.weights[type]
    
    def adam_aggregation(self, gradient_list: List[List[float]], type: str, lr: float=0.001, betas: Tuple[float, float]=(0.9, 0.999), eps: float=1e-8) -> List[float]:
        self._t_[type] += 1
        # get gradients
        self.gradients[type] = torch.tensor(gradient_list, dtype=torch.float32, requires_grad=False).mean(0)
        # get parameters
        parameters = torch.tensor(self.weights[type], dtype=torch.float32, requires_grad=False)
        # update biased first moment estimate
        self._m_[type] = betas[0] * self._m_[type] + (1 - betas[0]) * self.gradients[type]
        # update biased second raw moment estimate
        self._v_[type] = betas[1] * self._v_[type] + (1 - betas[1]) * self.gradients[type].square()
        # compute bias-corrected first moment estimate
        bias_corrected_m = self._m_[type] / (1 - betas[0] ** self._t_[type])
        # compute bias-corrected second raw moment estimate
        bias_corrected_v = self._v_[type] / (1 - betas[1] ** self._t_[type])
        # update parameters
        parameters -= lr * bias_corrected_m / (bias_corrected_v.sqrt() + eps)

        self.weights[type] = parameters.data.tolist()
        return self.weights[type]