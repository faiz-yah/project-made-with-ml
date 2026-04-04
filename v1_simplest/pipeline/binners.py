import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import Any

def get_binner(name: str, col: str = None, custom_bins: dict = None) -> Any:
    if name == "by_interval": 
        return KBinsDiscretizer(strategy="uniform", encode='ordinal')                              
    if name == "by_frq":      
        return KBinsDiscretizer(strategy="quantile", encode='ordinal')                             
    if name == "custom":                                                                                   
      if not custom_bins or col not in custom_bins:         
          raise ValueError(f"custom_bins missing entry for column '{col}'. Check your YAML.")
      preset = custom_bins[col]                                                                          
      return CustomBinarisation(bins=preset["bins"], labels=preset["labels"])
    raise ValueError(f"Binner not recognised: {name}")


## ----- Self-build  ----- ###

class CustomBinarisation():
    def __init__(self, bins, labels):
        self.bins = bins
        self.labels = labels
     
    def fit(self, X):
        return self
         
    def transform(self, X):
        result = pd.cut(X.squeeze(), bins=self.bins, labels=self.labels)                                   
        return result.cat.codes   # 'child'→0, 'teen'→1, 'adult'→2, 'senior'→3
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    