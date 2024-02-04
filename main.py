from utils.dataset import FLMNIST, mnist_data_split
from nodes.client import Client
from nodes.server import Server
from models.LeNet import LeNet

from torchvision import transforms
from torch.utils.data import DataLoader
from typing import List

import torch
from random import sample


num_client = 100
num_data_per_client = 512
non_iid_ratio = 0.1
client_sel_ratio = 0.4

num_of_test_data = 10000
total_rounds = 50
local_rounds = 4
train_batch_size = 128
test_batch_size = 256

if __name__ == "__main__":
    
    # split the data
    data_split = mnist_data_split(root="./datasets/", num_client=num_client, num_data_per_client=num_data_per_client, non_iid_ratio=non_iid_ratio)

    client_train_set = [
        FLMNIST(
            root="./datasets/",
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]),
            target_transform=None,
            download=False,
            data_ids=_ids) for _ids in data_split
    ]

    test_iter = DataLoader(
        dataset=FLMNIST(
            root="./datasets/",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]),
            target_transform=None,
            download=False,
            data_ids=range(num_of_test_data)),
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=0
    )

    # initialize clients and server
    model = LeNet()
    model_structure = [0]
    init_weights = []
    for param in model.parameters():
        model_structure.append(param.data.numel() + model_structure[-1])
        init_weights += param.data.cpu().view(-1).tolist()

    optimizer = torch.optim.Adagrad
    loss_func = torch.nn.CrossEntropyLoss()
    
    clients: List[Client] = [
        Client(dataset=client_train_set[i], model_structure=model_structure, device='cuda') for i in range(num_client)
    ]
    
    server = Server(model_structure=model_structure)
    server.set_weights(weights=init_weights)

    # locally alone training
    client = clients[0]
    client.set_weights(server.get_weights())
    for epoch in range(1, total_rounds+1):
        client.local_train_step(
            model=model, loss_func=loss_func, optimizer=optimizer, 
            batch_size=train_batch_size, local_rounds=1, num_workers=0, lr=0.01)
        acc = client.evaluate_accuracy(data_iter=test_iter, model=model, device='cuda')
        print("epoch-%d accuracy=%.3f"%(epoch, acc))

    # fl training
    for epoch in range(1, total_rounds+1):

        print("epoch:", epoch, "| gpu memory occupied:%.2f"%(torch.cuda.memory_allocated()/1024**3))
        weight_list = []

        sel_clients = sample(range(num_client), int(client_sel_ratio*num_client))
        
        for cid, client in enumerate(clients):
            if cid not in sel_clients:
                continue
            
            client.set_weights(server.get_weights())

            weight_list.append(
                client.local_train_step(
                    model=model, loss_func=loss_func, optimizer=optimizer,
                    local_rounds=local_rounds, batch_size=train_batch_size, num_workers=0, lr=0.01)
            )

        #     print(
        #         cid, client.evaluate_accuracy(test_iter, model, "cuda"), end="|"
        #     )
        # print()

        agg_weights = server.naive_aggregation(weight_list=weight_list)
        print(
            "server accuracy = %.3f"%server.evaluate_accuracy(test_iter, model, "cuda")
        )
        
        print()
