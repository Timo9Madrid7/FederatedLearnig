from torchvision.datasets import CIFAR10
from nodes.client import Client
from nodes.server import Server
from models.ResNet import resnet20 as ResNet

from torchvision import transforms
from torch.utils.data import DataLoader
from typing import List

import torch
from random import sample

from utils.dataset import FLClientDataset
from utils.drawer import Plot2D
from utils.model import save_model_state_dict
from utils.timer import get_timestamp

dataset_root = "./datasets/"
model_checkpoint = None
timestamp = get_timestamp()

num_client = 100
num_sample_per_client = 500
non_iid_ratio = 0.1
client_sel_ratio = 0.4

num_of_test_data = 10000
total_rounds = 250
local_rounds = 1
train_batch_size = 128
test_batch_size = 256

learning_rate = 0.01
download = False

if __name__ == "__main__":
    # download and initialize dataset
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    dataset = CIFAR10(root=dataset_root, train=True, transform=transform_train, target_transform=None, download=download)


    # data split
    flClientDataset = FLClientDataset(
        dataset=dataset, num_clients=num_client, num_sample_per_client=num_sample_per_client,
        non_iid_ratio=non_iid_ratio, unique_sample_sharing=True, shuffle=True, random_state=215
        )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_iter = DataLoader(
        dataset=CIFAR10(
            root=dataset_root,
            train=False,
            transform=transform_test,
            target_transform=None,
            download=download),
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=0
    )


    # initialize clients and server
    model = ResNet()
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))
    model_structure = [0]
    init_weights = []
    for param in model.parameters():
        model_structure.append(param.data.numel() + model_structure[-1])
        init_weights += param.data.cpu().view(-1).tolist()

    optimizer = torch.optim.Adagrad
    loss_func = torch.nn.CrossEntropyLoss()
    
    clients: List[Client] = [
        Client(dataset=flClientDataset.getClientDataset(i), model_structure=model_structure, device='cuda') for i in range(num_client)
    ]
    
    server = Server(model_structure=model_structure)
    server.set_weights(weights=init_weights)

    
    # initialize plot2D for plotting accuracy and loss
    plot2D = Plot2D()
    x_axis = list(range(1, total_rounds + 1))
    plot_prefix = f"./result/{timestamp}/" + __file__.split('/')[-1].split('.')[0]
 

    # locally alone training
    alone_loss_history = []
    alone_accuracy_history = []
    client = clients[0]
    client.set_weights(server.get_weights())
    for epoch in range(1, total_rounds+1):
        _, client_loss = client.local_train_step(
            model=model, loss_func=loss_func, optimizer=optimizer, 
            batch_size=train_batch_size, local_rounds=1, num_workers=0, lr=learning_rate)
        acc = client.evaluate_accuracy(data_iter=test_iter, model=model, device='cuda')
        print("epoch-%d accuracy=%.3f"%(epoch, acc))
        alone_loss_history.append(client_loss)
        alone_accuracy_history.append(acc)

    plot2D.plot([x_axis], [alone_loss_history], plot_prefix + "_alone_loss.png", nrows=1, ncols=1, xlabel="round", ylabel="loss")
    plot2D.plot([x_axis], [alone_accuracy_history], plot_prefix + "_alone_accuracy.png", nrows=1, ncols=1, xlabel="round", ylabel="accuracy")


    # fl training
    client_x_axis = [[] for _ in range(10)]
    fl_loss_history = [[] for _ in range(10)]  # record only for first 10 clients
    fl_acc_history = []                        # record the global accuracy
    for epoch in range(1, total_rounds+1):

        # print("epoch:", epoch, "| gpu memory occupied:%.2f"%(torch.cuda.memory_allocated()/1024**3))
        weight_list = []

        sel_clients = sample(range(num_client), int(client_sel_ratio*num_client))
        
        for cid, client in enumerate(clients):
            if cid not in sel_clients:
                continue
            
            client.set_weights(server.get_weights())

            client_weights, client_loss =  client.local_train_step(
                model=model, loss_func=loss_func, optimizer=optimizer,
                local_rounds=local_rounds, batch_size=train_batch_size, num_workers=0, lr=learning_rate)

            weight_list.append(client_weights)

            if 0 <= cid < 10:
                client_x_axis[cid].append(epoch)
                fl_loss_history[cid].append(client_loss)

        #     print(
        #         cid, client.evaluate_accuracy(test_iter, model, "cuda"), end="|"
        #     )
        # print()

        agg_weights = server.naive_aggregation(weight_list=weight_list)
        acc = server.evaluate_accuracy(test_iter, model, "cuda")
        print("epoch-%d accuracy=%.3f"%(epoch, acc))
        fl_acc_history.append(acc)

    plot2D.plot(client_x_axis, fl_loss_history, plot_prefix + "_fl_loss.png", figsize=[10, 5], nrows=2, ncols=5, xlabel="round", ylabel="loss")
    plot2D.plot([x_axis], [fl_acc_history], plot_prefix + "_fl_accuracy.png", nrows=1, ncols=1, xlabel="round", ylabel="accuracy")

    save_model_state_dict(model, f"./models/checkpoint/{timestamp}/" + __file__.split('/')[-1].split('.')[0])