import torch
from torch.utils.data import DataLoader
from torchsummary import summary, ModelStatistics
from typing import Tuple, Optional
from utils.drawer import TensorToImage

class GANTrainer:
    def __init__(self, dNet: torch.nn.Module, gNet: torch.nn.Module, dOptim: torch.optim.Optimizer, gOptim: torch.optim.Optimizer, criterion: torch.nn.Module) -> None:
        self._dNet_ = dNet
        self._gNet_ = gNet
        self._dOptim_ = dOptim
        self._gOptim_ = gOptim
        self._criterion_ = criterion

        self._device_ = next(self._dNet_.parameters()).device.type

    def get_discriminator_info(self, input_shape: Tuple[int], verbose: bool=False) -> ModelStatistics:
        return summary(self._dNet_, input_data=input_shape, verbose=verbose)
    
    def get_generator_info(self, input_shape: Tuple[int], verbose: bool=False) -> ModelStatistics:
        return summary(self._gNet_, input_data=input_shape, verbose=verbose)
    
    def __gen_noise__(self, batch_size: int, noise_dim: int) -> torch.Tensor:
        return torch.randn(batch_size, noise_dim, 1, 1, device=self._device_)
    
    def __gen_samples__(self, batch_szie: int, noise_dim: int) -> torch.Tensor:
        with torch.no_grad():
            noise = self.__gen_noise__(batch_size=batch_szie, noise_dim=noise_dim)
        return self._gNet_(noise)
    
    def __train_discriminator__(self, real_samples: torch.Tensor, noise_dim: int) -> float:
        self._dOptim_.zero_grad()

        noise_samples = self.__gen_noise__(batch_size=real_samples.shape[0], noise_dim=noise_dim)
        with torch.no_grad():
            fake_samples: torch.Tensor = self._gNet_(noise_samples)

        real_samples_pred = self._dNet_(real_samples)
        real_samples_loss = self._criterion_(real_samples_pred, torch.ones_like(real_samples_pred))
        fake_samples_pred = self._dNet_(fake_samples.detach())
        fake_samples_loss = self._criterion_(fake_samples_pred, torch.zeros_like(fake_samples_pred))
        loss: torch.Tensor = 0.5 * (real_samples_loss + fake_samples_loss)
        
        loss.backward()
        self._dOptim_.step()

        return loss.item()

    def __train_generator__(self, batch_size: int, noise_dim: int) -> float:
        self._gOptim_.zero_grad()
        
        noise_samples = self.__gen_noise__(batch_size=batch_size, noise_dim=noise_dim)
        fake_samples: torch.Tensor = self._gNet_(noise_samples)

        fake_samples_pred = self._dNet_(fake_samples)
        loss: torch.Tensor = self._criterion_(fake_samples_pred, torch.ones_like(fake_samples_pred))

        loss.backward()
        self._gOptim_.step()
        
        return loss.item()
    
    def train(self, dataloader: DataLoader, n_epochs: int, noise_dim: int, verbose: bool=True, converter: Optional[TensorToImage]=None, **converter_kwargs) -> None:
        d_loss, g_loss, n_steps = 0., 0., len(dataloader)
        for epoch in range(1, n_epochs + 1):
            for step, (real_samples, _) in enumerate(dataloader, 1):
                batch_size = real_samples.shape[0]
                real_samples = real_samples.to(self._device_)

                d_loss += self.__train_discriminator__(real_samples=real_samples, noise_dim=noise_dim)
                g_loss += self.__train_generator__(batch_size=batch_size, noise_dim=noise_dim)

            if verbose:
                print("epoch: [%d/%d] | d_loss: %.3f | g_loss: %.3f"%(epoch, n_epochs, d_loss / n_steps, g_loss / n_steps))
                d_loss, g_loss = 0., 0.

            if converter:
                tensors = self.__gen_samples__(batch_szie=converter_kwargs["n_samples"], noise_dim=noise_dim)
                img = converter.convert(tensor=tensors, n_samples=converter_kwargs["n_samples"], n_rows=converter_kwargs["n_rows"])
                img.save(converter_kwargs['path'] + f"epoch_{epoch}.png")