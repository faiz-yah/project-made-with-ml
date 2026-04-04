import pandas as pd
import numpy as np
from typing import Protocol, runtime_checkable
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, TargetEncoder
from typing import Any


def get_encoder(name:str) -> Any:
    if name == 'ordinal':
        return OrdinalEncoder()
    if name == 'onehot':
        return OneHotEncoder()
    if name == 'frequency':
        return FrequencyEncoder()
    if name == 'target':
        return TargetEncoder()
    else:
        raise ValueError(f"Unknown encoder: {name}")


## ----- Self-build  ----- ###


class FrequencyEncoder():
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.mapping_dict_ = X.squeeze().value_counts(dropna=False).to_dict()
        return self
    
    def transform(self, X):
        return X.squeeze().map(self.mapping_dict_)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)