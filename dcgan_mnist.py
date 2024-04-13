import torch
from models.GAN import DCDiscriminator, DCGenerator
from algorithms.gan import GANTrainer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from utils.timer import get_timestamp
from utils.drawer import TensorToImage
from os import makedirs

dataset_root = "./datasets/"
download = False
timestamp = get_timestamp()

batch_size = 64
learning_rate = 0.0002
device = 'cuda'
n_epochs = 100

ndf, ngf, nc, nz = 32, 32, 1, 100
image_size = (1, 28, 28)
mode = None

if __name__ == "__main__":
    dataloader = DataLoader(
        MNIST("./datasets", train=True, download=download, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    dNet = DCDiscriminator(ndf=ndf, nc=nc).to(device=device)
    gNet = DCGenerator(nz=nz, ngf=ngf, nc=nc).to(device=device)

    dOptim = torch.optim.Adam(dNet.parameters(), lr=learning_rate)
    gOptim = torch.optim.Adam(gNet.parameters(), lr=learning_rate)

    criterion = torch.nn.BCELoss()

    converter = TensorToImage(image_size=image_size, mode=mode)
    path = f"./result/{timestamp}/"
    makedirs(name=path, exist_ok=True)
    converter_config = {"noise": torch.randn(30, 100, 1, 1, device=device), "n_samples": 30, "n_rows": 6, "path": path}

    ganTrainer = GANTrainer(dNet=dNet, gNet=gNet, dOptim=dOptim, gOptim=gOptim, criterion=criterion)
    ganTrainer.train(dataloader=dataloader, n_epochs=n_epochs, noise_dim=nz, verbose=True, converter=converter, **converter_config)
    converter.toGIF(path, remove_cache=True, fps=2.5, loop=1)