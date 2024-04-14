import torch
from models.GAN import cDCDiscriminator, cDCGenerator
from algorithms.gan import cGANTrainer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from utils.timer import get_timestamp
from utils.drawer import TensorToImage
from utils.model import init_conv_weights
from os import makedirs

dataset_root = "./datasets/"
download = False
timestamp = get_timestamp()

batch_size = 64
learning_rate = 0.0002
betas = (0.5, 0.999)
lr_decay_step_size = 5
lr_decay_gamma = 0.5
device = 'cuda'
n_epochs = 20

ndf, ngf, nc, nz = 64, 128, 1, 100
image_size = (1, 32, 32)
num_class = 10
mode = None

if __name__ == "__main__":
    dataloader = DataLoader(
        MNIST("./datasets", train=True, download=download, transform=transforms.Compose([
            transforms.Resize(image_size[1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5, ))])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    dNet = cDCDiscriminator(ndf=ndf, nc=nc, num_class=num_class).to(device=device)
    gNet = cDCGenerator(nz=nz, ngf=ngf, nc=nc, num_class=num_class).to(device=device)
    init_conv_weights(dNet)
    init_conv_weights(gNet)

    dOptim = torch.optim.Adam(dNet.parameters(), lr=learning_rate, betas=betas)
    gOptim = torch.optim.Adam(gNet.parameters(), lr=learning_rate, betas=betas)
    dLrScheduler = torch.optim.lr_scheduler.StepLR(dOptim, step_size=lr_decay_step_size, gamma=lr_decay_gamma)
    gLrScheduler = torch.optim.lr_scheduler.StepLR(gOptim, step_size=lr_decay_step_size, gamma=lr_decay_gamma)

    criterion = torch.nn.BCELoss()

    converter = TensorToImage(image_size=image_size, mode=mode)
    path = f"./result/{timestamp}/"
    makedirs(name=path, exist_ok=True)
    converter_config = {
        "noise": torch.randn(num_class * num_class, nz, 1, 1, device=device), 
        "labels": torch.LongTensor([i % num_class for i in range(num_class * num_class)]),              
        "n_samples": num_class * num_class, "n_rows": num_class, "path": path}

    ganTrainer = cGANTrainer(dNet=dNet, gNet=gNet, dOptim=dOptim, gOptim=gOptim, dLrScheduler=dLrScheduler, gLrScheduler=gLrScheduler, criterion=criterion, num_class=num_class, image_size=image_size[1])
    ganTrainer.train(dataloader=dataloader, n_epochs=n_epochs, noise_dim=nz, verbose=True, converter=converter, **converter_config)
    converter.toGIF(path, remove_cache=True, fps=1, loop=1)