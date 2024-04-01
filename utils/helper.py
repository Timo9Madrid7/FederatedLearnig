from functools import wraps
from typing import List
from collections import defaultdict

def fl_getitem(f):
    """
    override __getitem__ 

    example:
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
    """

    @wraps(f)
    
    def decorated(cls, *args, **kwargs):
        for key, value in kwargs.items():
            kwargs[key] = cls.data_ids[value]

        for arg in args:
            arg = cls.data_ids[arg]

        return f(cls, *args, **kwargs)
    
    return decorated


    


    