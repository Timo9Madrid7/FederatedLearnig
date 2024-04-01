from typing import List, Tuple
from torch.utils.data import Dataset
from nodes.node import Node

import torch
from torch.utils.data import DataLoader

class Client(Node):

    def __init__(self, dataset: Dataset, model_structure: List, device: str) -> None:
        super().__init__(model_structure)

        self.dataset = dataset
        self.device = device
    
    def local_train_step(self, model: torch.nn.Module, loss_func: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_size: int, local_rounds: int = 1, num_workers: int = 0, **optimizer_settings) -> Tuple[List[float], float]:
        # model loading
        for layer, param in enumerate(model.parameters(), 1):
            param.data = torch.tensor(self.weights[self.model_structure[layer-1]:self.model_structure[layer]]).view(param.data.size())
        model = model.to(self.device)
        
        # optimizer configuration
        model_optimizer: torch.optim.Optimizer = optimizer(model.parameters(), **optimizer_settings)

        # data loading
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # train step
        model.train()
        avg_loss = 0
        for _ in range(local_rounds):
            for batch_idx, (inputs, targets) in enumerate(dataloader, 1):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                model_optimizer.zero_grad()
                outputs = model(inputs)
                loss: torch.Tensor = loss_func(outputs, targets)
                avg_loss += float(loss.detach().cpu())
                loss.backward()
                model_optimizer.step()
        avg_loss /= local_rounds * batch_idx

        # local model save
        self.weights = []
        for param in model.parameters():
            self.weights += param.data.cpu().view(-1).tolist()

        return self.weights, avg_loss