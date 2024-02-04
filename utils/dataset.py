from typing import Tuple, List, Any
from typing_extensions import override
from torchvision.datasets import MNIST
from collections import defaultdict

from utils.helper import fl_getitem

class FLMNIST(MNIST):

    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False, data_ids: list = range(60000)) -> None:
        super().__init__(root, train, transform, target_transform, download)

        self.data_ids = data_ids

    @override
    def __len__(self):
        return len(self.data_ids)

    @override
    @fl_getitem
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)
    
def mnist_data_split(root: str, num_client: int, num_data_per_client: int, non_iid_ratio: int=0.1) -> List[List[int]]:
    assert num_client * num_data_per_client <= 60000
    
    dataset = FLMNIST(root=root, train=True, download=False)

    indices_by_class = defaultdict(list)
    for _ids in dataset.data_ids:
        indices_by_class[dataset.__getitem__(_ids)[1]].append(_ids)

    main_class_number = int(num_data_per_client * non_iid_ratio)
    other_class_number = (num_data_per_client - main_class_number) // 9

    res = [[] for _ in range(num_client)]
    p = [0 for _ in range(10)]
    for client_id in range(num_client):
        for i in range(10):
            num_data = main_class_number if client_id % 10 == i else other_class_number
            res[client_id] += indices_by_class[i][p[i] : p[i] + num_data]
            p[i] += num_data

    return res

        