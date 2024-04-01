from torch.utils.data.dataset import Dataset
from collections import defaultdict
from torch.utils.data import Subset
from numpy.random import RandomState
from typing import List

class FLClientDataset():
    def __init__(self, dataset: Dataset, num_clients: int, num_sample_per_client: int, non_iid_ratio: int, unique_sample_sharing: bool=True, *args, **kwargs) -> None:
        self._num_clients_ = num_clients
        self._num_sample_per_client_ = num_sample_per_client
        self._non_iid_ratio_ = non_iid_ratio
        self._unique_sample_sharing_ = unique_sample_sharing

        self._dataset_ = dataset
        self._args_ = args
        self._kwargs_ = kwargs

        self.__client_sample_id__ = [[] for _ in range(self._num_clients_)]
        self.__share_data__(self._kwargs_['shuffle'], self._kwargs_['random_state'])


    def __share_data__(self, shuffle: bool=True, random_state: int=215):
        if self._unique_sample_sharing_ and self._num_clients_ * self._num_sample_per_client_ > len(self._dataset_):
            raise IndexError("No enough samples for splitting")
        
        indices_by_class = defaultdict(list)
        for idx, (_, y) in enumerate(self._dataset_):
            indices_by_class[y].append(idx)
        num_classes = len(indices_by_class)

        random = RandomState(random_state)
        if shuffle:
            for class_id in indices_by_class.keys():
                random.shuffle(indices_by_class[class_id])            
        
        main_class_number = int(self._num_sample_per_client_ * self._non_iid_ratio_)
        other_class_number = (self._num_sample_per_client_ - main_class_number) // (num_classes - 1)

        if self._unique_sample_sharing_:
            p = [0 for _ in range(num_classes)]
            for client_id in range(self._num_clients_):
                for class_id in indices_by_class.keys():
                    num_data = main_class_number if client_id % num_classes == class_id else other_class_number
                    self.__client_sample_id__[client_id].extend(indices_by_class[class_id][p[class_id] : p[class_id] + num_data])
                    p[class_id] += num_data
        
        else:
            try:
                for client_id in range(self._num_clients_):
                    for class_id in indices_by_class.keys():
                        num_data = main_class_number if client_id % num_classes == class_id else other_class_number
                        self.__client_sample_id__[client_id].extend(random.choice(indices_by_class[class_id], num_data, replace=False).tolist())
            except ValueError:
                raise IndexError("No enough samples for splitting")
            

    def getClientDataset(self, idx: int) -> Subset:
        return Subset(self._dataset_, self.__client_sample_id__[idx])
    
    
    def getClientSampleId(self, idx: int) -> List[int]:
        return self.__client_sample_id__[idx]