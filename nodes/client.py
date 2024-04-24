from typing import List, Tuple, Dict
import torch
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from nodes.node import Node, GenNode


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
    

class GenClient(GenNode):
    def __init__(self, dataset: Dataset, model_structure: Dict[str, List[int]], device: str) -> None:
        super().__init__(model_structure)
        
        self.dataset = dataset
        self.device = device

    def __gen_noise__(self, batch_size: int, noise_dim: int) -> torch.Tensor:
        return torch.randn(batch_size, noise_dim, 1, 1, device=self.device)
    
    def local_train_step(self, model: Dict[str, torch.nn.Module], loss_func: torch.nn.Module, optimizer: Dict[str, torch.optim.Optimizer], batch_size: int, local_rounds: int = 1, noise_dim: int = 100, num_workers: int = 0, **optimizer_settings) -> Tuple[List[float], List[float], List[float], List[float], float, float]:
        model_optimizer: Dict[str, torch.optim.Optimizer] = {}
        for type in ('d', 'g'):
            # model loading
            for layer, param in enumerate(model[type].parameters(), 1):
                param.data = torch.tensor(self.weights[type][self.model_structure[type][layer-1]:self.model_structure[type][layer]]).view(param.data.size())
            model[type] = model[type].to(self.device)
            # optimizer configuration
            model_optimizer[type] = optimizer[type](model[type].parameters(), **optimizer_settings)

        # data loading
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # sampler = RandomSampler(self.dataset, num_samples=batch_size)
        # batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        # dataloader = DataLoader(self.dataset, batch_sampler=batch_sampler, num_workers=num_workers)

        # train step
        d_loss, g_loss = 0., 0.
        for _ in range(local_rounds):
            for batch_idx, (real_samples, _) in enumerate(dataloader, 1):
                ### Train Discriminator ###
                batch_size = real_samples.shape[0]
                real_samples = real_samples.to(self.device)

                model_optimizer['d'].zero_grad()
                
                with torch.no_grad():
                    noise_samples = self.__gen_noise__(batch_size=batch_size, noise_dim=noise_dim)
                    fake_samples: torch.Tensor = model['g'](noise_samples)
                
                real_samples_pred = model['d'](real_samples)
                real_samples_loss = loss_func(real_samples_pred, torch.ones_like(real_samples_pred))
                fake_samples_pred = model['d'](fake_samples.detach())
                fake_samples_loss = loss_func(fake_samples_pred, torch.zeros_like(fake_samples_pred))
                loss: torch.Tensor = 0.5 * (real_samples_loss + fake_samples_loss)

                loss.backward()
                model_optimizer['d'].step()

                d_loss += loss.item()
                ##############################################################################################

                ### Train Generator ###
                model_optimizer['g'].zero_grad()

                noise_samples = self.__gen_noise__(batch_size=batch_size, noise_dim=noise_dim)
                fake_samples: torch.Tensor = model['g'](noise_samples)

                fake_samples_pred = model['d'](fake_samples)
                loss: torch.Tensor = loss_func(fake_samples_pred, torch.ones_like(fake_samples_pred))

                loss.backward()
                model_optimizer['g'].step()

                g_loss += loss.item()
                ##############################################################################################

        d_loss /= (batch_idx * local_rounds)
        g_loss /= (batch_idx * local_rounds)
        
        # local model save
        for type in ('d', 'g'):
            self.weights[type] = []
            self.gradients[type] = []
            for param in model[type].parameters():
                self.weights[type] += param.data.cpu().view(-1).tolist()
                self.gradients[type] += param.grad.data.cpu().view(-1).tolist() 

        return self.weights['d'], self.weights['g'], self.gradients['d'], self.gradients['g'], d_loss, g_loss
    
    def local_train_step_by_type(self, train_type: str, model: Dict[str, torch.nn.Module], loss_func: torch.nn.Module, optimizer: Dict[str, torch.optim.Optimizer], batch_size: int, local_rounds: int = 1, noise_dim: int = 100, num_workers: int = 0, **optimizer_settings) -> Tuple[List[float], List[float], float]:
        model_optimizer: Dict[str, torch.optim.Optimizer] = {}
        for type in ('d', 'g'):
            # model loading
            for layer, param in enumerate(model[type].parameters(), 1):
                param.data = torch.tensor(self.weights[type][self.model_structure[type][layer-1]:self.model_structure[type][layer]]).view(param.data.size())
            model[type] = model[type].to(self.device)
            # optimizer configuration
            model_optimizer[type] = optimizer[type](model[type].parameters(), **optimizer_settings)

        # data loading
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # train step
        d_loss, g_loss = 0., 0.
        for _ in range(local_rounds):
            for batch_idx, (real_samples, _) in enumerate(dataloader, 1):
                batch_size = real_samples.shape[0]

                if train_type == 'd':
                    real_samples = real_samples.to(self.device)

                    model_optimizer['d'].zero_grad()

                    with torch.no_grad():
                        noise_samples = self.__gen_noise__(batch_size=batch_size, noise_dim=noise_dim)
                        fake_samples: torch.Tensor = model['g'](noise_samples)
                    
                    real_samples_pred = model['d'](real_samples)
                    real_samples_loss = loss_func(real_samples_pred, torch.ones_like(real_samples_pred))
                    fake_samples_pred = model['d'](fake_samples.detach())
                    fake_samples_loss = loss_func(fake_samples_pred, torch.zeros_like(fake_samples_pred))
                    loss: torch.Tensor = 0.5 * (real_samples_loss + fake_samples_loss)

                    loss.backward()
                    model_optimizer['d'].step()

                    d_loss += loss.item()

                elif train_type == 'g':
                    model_optimizer['g'].zero_grad()

                    noise_samples = self.__gen_noise__(batch_size=batch_size, noise_dim=noise_dim)
                    fake_samples: torch.Tensor = model['g'](noise_samples)
                    fake_samples: torch.Tensor = model['g'](noise_samples)

                    fake_samples_pred = model['d'](fake_samples)
                    loss: torch.Tensor = loss_func(fake_samples_pred, torch.ones_like(fake_samples_pred))

                    loss.backward()
                    model_optimizer['g'].step()

                    g_loss += loss.item()
                    
        d_loss /= (batch_idx * local_rounds)
        g_loss /= (batch_idx * local_rounds)
        
        # local model save
        self.weights[train_type] = []
        self.gradients[train_type] = []
        for param in model[train_type].parameters():
            self.weights[train_type] += param.data.cpu().view(-1).tolist()
            self.gradients[train_type] += param.grad.data.cpu().view(-1).tolist() 

        return self.weights[train_type], self.gradients[train_type], d_loss if train_type == 'd' else g_loss