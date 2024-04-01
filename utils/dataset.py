from torch.utils.data.dataset import Dataset
from collections import defaultdict
from torch.utils.data import Subset
from numpy.random import RandomState
from typing import List

class FLClientDataset():
    def __init__(self, dataset: Dataset, num_clients: int, num_sample_per_client: int, non_iid_ratio: int, unique_sample_sharing: bool=True, *args, **kwargs) -> None:
        self.num_clients = num_clients
        self.num_sample_per_client = num_sample_per_client
        self.non_iid_ratio = non_iid_ratio
        self.unique_sample_sharing = unique_sample_sharing

        self.dataset = dataset
        self.args = args
        self.kwargs = kwargs

        self.__client_sample_id__ = [[] for _ in range(self.num_clients)]
        self.__share_data__(self.kwargs['shuffle'], self.kwargs['random_state'])


    def __share_data__(self, shuffle: bool=True, random_state: int=215):
        if self.unique_sample_sharing and self.num_clients * self.num_sample_per_client > len(self.dataset):
            raise IndexError("No enough samples for splitting")
        
        indices_by_class = defaultdict(list)
        for idx, (_, y) in enumerate(self.dataset):
            indices_by_class[y].append(idx)
        num_classes = len(indices_by_class)

        random = RandomState(random_state)
        if shuffle:
            for class_id in indices_by_class.keys():
                random.shuffle(indices_by_class[class_id])            
        
        main_class_number = int(self.num_sample_per_client * self.non_iid_ratio)
        other_class_number = (self.num_sample_per_client - main_class_number) // (num_classes - 1)

        if self.unique_sample_sharing:
            p = [0 for _ in range(num_classes)]
            for client_id in range(self.num_clients):
                for class_id in indices_by_class.keys():
                    num_data = main_class_number if client_id % num_classes == class_id else other_class_number
                    self.__client_sample_id__[client_id].extend(indices_by_class[class_id][p[class_id] : p[class_id] + num_data])
                    p[class_id] += num_data
        
        else:
            try:
                for client_id in range(self.num_clients):
                    for class_id in indices_by_class.keys():
                        num_data = main_class_number if client_id % num_classes == class_id else other_class_number
                        self.__client_sample_id__[client_id].extend(random.choice(indices_by_class[class_id], num_data, replace=False).tolist())
            except ValueError:
                raise IndexError("No enough samples for splitting")
            

    def getClientDataset(self, idx: int) -> Subset:
        return Subset(self.dataset, self.__client_sample_id__[idx])
    
    
    def getClientSampleId(self, idx: int) -> List[int]:
        return self.__client_sample_id__[idx]