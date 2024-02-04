from functools import wraps
from typing import List
from collections import defaultdict

def fl_getitem(f):
    """
    override __getitem__ 
    """

    @wraps(f)
    
    def decorated(cls, *args, **kwargs):
        for key, value in kwargs.items():
            kwargs[key] = cls.data_ids[value]

        for arg in args:
            arg = cls.data_ids[arg]

        return f(cls, *args, **kwargs)
    
    return decorated


    


    