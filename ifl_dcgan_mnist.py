
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Dict, List
from random import sample
from os import makedirs

from utils.dataset import FLClientDataset
from nodes.client import GenClient
from nodes.server import GenServer
from models.GAN import DCGenerator, DCDiscriminator

from utils.model import init_conv_weights, cal_mmd
from utils.timer import get_timestamp
from utils.drawer import TensorToImage

dataset_root = "./datasets/"
download = False
timestamp = get_timestamp()

num_client = 100
num_sample_per_client = 512
non_iid_ratio = 0.1
client_sel_ratio = 0.4

total_rounds = 200
local_rounds = 1
train_batch_size = 64
learning_rate = 0.0002
betas = (0.5, 0.999)
device = 'cuda'
num_workers = 0

ndf, ngf, nc, nz = 32, 32, 1, 100
image_size = 28
mode = None

standalone_compare = False

if __name__ == '__main__':
    # visualization tool initialization
    tensor2img = TensorToImage((nc, image_size, image_size), mode=mode)
    path = f"./result/{timestamp}/"
    makedirs(name=path, exist_ok=True)
    fixed_noise = torch.randn(100, nz, 1, 1, device=device)

    # download and initialize dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    dataset = MNIST(root=dataset_root, train=True, transform=transform, target_transform=None, download=download)

    # data split
    flClientDataset = FLClientDataset(
        dataset=dataset, num_clients=num_client, num_sample_per_client=num_sample_per_client,
        non_iid_ratio=non_iid_ratio, unique_sample_sharing=True, shuffle=True, random_state=215
        )
    
    # initialize clients and server
    model: Dict[str, torch.nn.Module]  = {'d': DCDiscriminator(ndf=ndf, nc=nc), 'g': DCGenerator(nz=nz, ngf=ngf, nc=nc)}
    init_conv_weights(model['d'])
    init_conv_weights(model['g'])

    model_structure: Dict[str, List[int]] = {'d': [0], 'g': [0]}
    init_weights: Dict[str, List[int]] = {'d': [], 'g': []}

    for m in ('d', 'g'):
        for param in model[m].parameters():
            model_structure[m].append(param.data.numel() + model_structure[m][-1])
            init_weights[m] += param.data.cpu().view(-1).tolist()

    optimizer: Dict[str, torch.optim.Optimizer] = {'d': torch.optim.Adam, 'g': torch.optim.Adam}
    loss_func = torch.nn.BCELoss()

    clients: List[GenClient] = [
        GenClient(dataset=flClientDataset.getClientDataset(i), model_structure=model_structure, device=device) for i in range(num_client)
    ]
    server = GenServer(model_structure=model_structure)
    server.set_weights(weights=init_weights)

    ### Standalone Training ###
    if standalone_compare:
        client = clients[0]
        client.set_weights(server.get_weights())
        for epoch in range(1, 10 * total_rounds + 1):
            _, _, _, _, dloss, gloss = client.local_train_step(
                model=model, loss_func=loss_func, optimizer=optimizer,
                batch_size=train_batch_size, num_workers=num_workers,
                lr=learning_rate, betas=betas,
            )

            print("epoch: [%d/%d] | d_loss: %.3f | g_loss: %.3f"%(epoch, 10 * total_rounds, dloss, gloss))
            
            # visualization
            if epoch % 10 == 0:
                with torch.no_grad():
                    model['g'] = model['g'].to(device)
                    gen_samples = model['g'](fixed_noise)
                img = tensor2img.convert(gen_samples, gen_samples.shape[0], 10)
                img.save(path + f"epoch_{epoch}.png")
        tensor2img.toGIF(path, remove_cache=True, gif_name="standalone", fps=1, loop=1)
    ########################################################################################################
    

    ### Federated Training TYPE 1 ###
    for epoch in range(1, total_rounds + 1):
        sel_clients = sample(range(num_client), int(client_sel_ratio * num_client))

        dweight_list, gweight_list, dgrad_list, ggard_list, mmd_list, dloss_sum, gloss_sum = [], [], [], [], [], 0., 0.
        for cid, client in enumerate(clients):
            if cid not in sel_clients:
                continue
            
            client.set_weights(server.get_weights())

            client_dweight, client_gweight, client_dgrad, client_ggrad, client_dloss, client_gloss = client.local_train_step(
                model=model, loss_func=loss_func, optimizer=optimizer,
                batch_size=train_batch_size, num_workers=num_workers,
                lr=learning_rate, betas=betas,
            )
            client_mmd = cal_mmd(model=model['g'], dataset=client.dataset)

            # dgrad_list.append(client_dgrad)
            # ggard_list.append(client_ggrad)
            mmd_list.append(client_mmd)
            dweight_list.append(client_dweight)
            gweight_list.append(client_gweight)
            dloss_sum += client_dloss
            gloss_sum += client_gloss

        server.naive_aggregation(weight_list=dweight_list, type='d')
        server.weighted_aggregation(weight_list=gweight_list, coef_list=mmd_list, type='g', softmax=True)
        print("epoch: [%d/%d] | d_loss: %.3f | g_loss: %.3f | mmd_min: %.3f | _max: %.3f"%(epoch, total_rounds, dloss_sum/len(sel_clients), gloss_sum/len(sel_clients), min(mmd_list), max(mmd_list)))
        
        # visualization
        intermediate_weights= server.get_weights()
        for layer, param in enumerate(model['g'].parameters(), 1):
            param.data = torch.tensor(intermediate_weights['g'][model_structure['g'][layer-1]:model_structure['g'][layer]]).view(param.data.size())
        with torch.no_grad():
            model['g'] = model['g'].to(device)
            gen_samples = model['g'](fixed_noise)
        img = tensor2img.convert(gen_samples, gen_samples.shape[0], 10)
        img.save(path + f"epoch_{epoch}.png")
    
    tensor2img.toGIF(path, remove_cache=True, gif_name="federated", fps=1, loop=1)
    ########################################################################################################