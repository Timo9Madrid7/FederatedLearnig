# reference from https://github.com/easezyc/deep-transfer-learning/blob/5e94d519b7bb7f94f0e43687aa4663aca18357de/MUDA/MFSAN/MFSAN_3src/mmd.py
import torch

def guassian_kernel(source: torch.Tensor, target: torch.Tensor, kernel_mul: float=2.0, kernel_num: int=5, fix_sigma: float=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source: torch.Tensor, target: torch.Tensor, kernel_mul: float=2.0, kernel_num: int=5, fix_sigma: float=None):
    """to calculate the Maximum Mean Discrepancy (MMD)
    """
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

if __name__ == '__main__':
    data_0 = torch.normal(mean=0, std=1, size=(32, 784))
    data_1 = torch.normal(mean=0, std=1, size=(32, 784))
    data_2 = torch.normal(mean=1, std=1, size=(32, 784))
    data_3 = torch.normal(mean=2, std=1, size=(32, 784))

    print("data 0 vs 1", mmd(data_0, data_1))
    print("data 1 vs 2", mmd(data_1, data_2))
    print("data 1 vs 3", mmd(data_1, data_3))
    print("data 2 vs 2", mmd(data_2, data_3))