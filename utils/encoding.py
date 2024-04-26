import torch

class Encoder:
    def __init__(self, num_class, image_size) -> None:
        self._num_class_ = num_class
        self._image_size_ = image_size
    
    def encode(self):
        pass

class OneHotEncoder4Gan(Encoder):
    def __init__(self, num_class, image_size) -> None:
        super().__init__(num_class, image_size)
        
        self._g_onehot_ = torch.zeros(self._num_class_, self._num_class_, 1, 1)
        self._g_onehot_ = self._g_onehot_.scatter_(dim=1, index=torch.arange(self._num_class_).view(self._num_class_, 1, 1, 1), value=1.)
        self._d_onehot_ = torch.zeros(self._num_class_, self._num_class_, self._image_size_, self._image_size_)
        for i in range(self._num_class_):
            self._d_onehot_[i, i].fill_(1.)

    def encode(self, labels: torch.Tensor, type: str) -> torch.Tensor:
        device = labels.device.type
        return self._g_onehot_.to(device)[labels].to(device) if type == 'g' else self._d_onehot_.to(device)[labels].to(device)
    
if __name__ == '__main__':
    onehotEncoder = OneHotEncoder4Gan(num_class=10, image_size=32)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long, requires_grad=False)
    d_labels_enc = onehotEncoder.encode(labels=labels, type='d')
    g_labels_enc = onehotEncoder.encode(labels=labels, type='g')