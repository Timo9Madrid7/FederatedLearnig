from typing import List
from nodes.node import Node
import torch

class Server(Node):

    def __init__(self, model_structure: List) -> None:
        super().__init__(model_structure)

    def naive_aggregation(self, weight_list: List[List[int]]) -> List[int]:
        weight_list : torch.Tensor = torch.tensor(weight_list, dtype=torch.float32, requires_grad=False)
        self.weights = weight_list.mean(0).data.tolist()
        return self.weights