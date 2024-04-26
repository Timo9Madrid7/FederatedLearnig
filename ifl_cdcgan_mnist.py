
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Dict, List
from random import sample
from os import makedirs

from utils.dataset import FLClientDataset
from nodes.client import GenClient
from nodes.server import GenServer
from models.GAN import cDCGenerator, cDCDiscriminator

from utils.model import init_conv_weights, cal_mmd
from utils.timer import get_timestamp, Timer
from utils.drawer import TensorToImage
from utils.encoding import OneHotEncoder4Gan

dataset_root = "./datasets/"
download = False
timestamp = get_timestamp()

num_client = 100
num_sample_per_client = 512
non_iid_ratio = 0.1
client_sel_ratio = 0.4

total_rounds = 40
local_rounds = 1
train_batch_size = 64
learning_rate = 0.0002
betas = (0.5, 0.999)
lr_decay_step_size = 5
lr_decay_gamma = 0.5
device = 'cuda'
num_workers = 0

ndf, ngf, nc, nz = 64, 128, 1, 100
image_size = 32
num_class = 10
mode = None

standalone_compare = False

if __name__ == '__main__':
    # initialize one-hot label encoder
    encoder = OneHotEncoder4Gan(num_class=num_class, image_size=image_size)

    # initialize timer
    timer = Timer()

    # visualization tool initialization
    tensor2img = TensorToImage((nc, image_size, image_size), mode=mode)
    path = f"./result/{timestamp}/"
    makedirs(name=path, exist_ok=True)
    fixed_noise = torch.randn(100, nz, 1, 1, device=device)
    fixed_labels = torch.tensor([i % num_class for i in range(num_class * num_class)], dtype=torch.long, device=device)
    fixed_labels = encoder.encode(fixed_labels, type='g')

    # download and initialize dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),
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
    model: Dict[str, torch.nn.Module]  = {'d': cDCDiscriminator(ndf=ndf, nc=nc, num_class=num_class), 'g': cDCGenerator(nz=nz, ngf=ngf, nc=nc, num_class=num_class)}
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
            _, _, _, _, dloss, gloss = client.local_train_cgan_step(
                model=model, loss_func=loss_func, optimizer=optimizer,  # model
                encoder=encoder,                                        # encoder
                batch_size=train_batch_size, num_workers=num_workers,   # learning
                lr=learning_rate, betas=betas,
            )

            print("epoch: [%d/%d] | d_loss: %.3f | g_loss: %.3f"%(epoch, 10 * total_rounds, dloss, gloss))
            
            # visualization
            if epoch % 10 == 0:
                with torch.no_grad():
                    model['g'] = model['g'].to(device)
                    gen_samples = model['g'](fixed_noise, fixed_labels)
                img = tensor2img.convert(gen_samples, gen_samples.shape[0], num_class)
                img.save(path + f"epoch_{epoch}.png")
        tensor2img.toGIF(path, remove_cache=True, gif_name="standalone", fps=1, loop=1)
    ########################################################################################################
    

    ### Federated Training TYPE 1 ###
    for epoch in range(1, total_rounds + 1):
        # learning rate strategy
        if epoch % lr_decay_step_size == 0:
            learning_rate *= lr_decay_gamma
        
        sel_clients = sample(range(num_client), int(client_sel_ratio * num_client))

        dweight_list, gweight_list, dgrad_list, ggard_list, mmd_list, dloss_sum, gloss_sum = [], [], [], [], [], 0., 0.
        # dweight_tensor, gweight_tensor = torch.zeros((model_structure['d'][-1], ), dtype=torch.float32, requires_grad=False), torch.zeros((model_structure['g'][-1], ), dtype=torch.float32, requires_grad=False)
        timer.start()
        for cid, client in enumerate(clients):
            if cid not in sel_clients:
                continue
            
            client.set_weights(server.get_weights())

            client_dweight, client_gweight, client_dgrad, client_ggrad, client_dloss, client_gloss = client.local_train_cgan_step(
                model=model, loss_func=loss_func, optimizer=optimizer,  # model
                encoder=encoder,                                        # encoder
                batch_size=train_batch_size, num_workers=num_workers,   # learning
                lr=learning_rate, betas=betas,
            )
            client_mmd = cal_mmd(model['g'], client.dataset, encoder=encoder)

            # dgrad_list.append(client_dgrad)
            # ggard_list.append(client_ggrad)
            mmd_list.append(client_mmd)
            dweight_list.append(client_dweight)
            gweight_list.append(client_gweight)
            # dweight_tensor += torch.tensor(client_dweight, dtype=torch.float32, requires_grad=False)
            # gweight_tensor += torch.tensor(client_gweight, dtype=torch.float32, requires_grad=False)
            dloss_sum += client_dloss
            gloss_sum += client_gloss
            client.reset_all() # for saving memory

        server.naive_aggregation(weight_list=dweight_list, type='d')
        # server.naive_aggregation(weight_list=gweight_list, type='g')
        server.weighted_aggregation(weight_list=gweight_list, coef_list=mmd_list, type='g', softmax=True)
        # server.set_weights(dweight_tensor.div(len(sel_clients)).tolist(), type='d')
        # server.set_weights(gweight_tensor.div(len(sel_clients)).tolist(), type='g')
        timer.end()
        _, mins, secs = timer.timeslot()
        print("epoch: [%d/%d] | d_loss: %.3f | g_loss: %.3f | mmd_min: %.3f | _max: %.3f | time: %dm%ds"%(epoch, total_rounds, dloss_sum/len(sel_clients), gloss_sum/len(sel_clients), min(mmd_list), max(mmd_list), mins, secs))

        # visualization
        intermediate_weights= server.get_weights()
        for layer, param in enumerate(model['g'].parameters(), 1):
            param.data = torch.tensor(intermediate_weights['g'][model_structure['g'][layer-1]:model_structure['g'][layer]]).view(param.data.size())
        with torch.no_grad():
            model['g'] = model['g'].to(device)
            gen_samples = model['g'](fixed_noise, fixed_labels)
        img = tensor2img.convert(gen_samples, gen_samples.shape[0], num_class)
        img.save(path + f"epoch_{epoch}.png")
    
    tensor2img.toGIF(path, remove_cache=True, gif_name="federated", fps=1, loop=1)
    ########################################################################################################