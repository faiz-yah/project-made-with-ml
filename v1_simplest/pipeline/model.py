from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 


def get_model(name:str, params:dict=None) -> Any:
    params = params or {}
    if name == 'logreg':
        clf = LogisticRegression(**params)
    elif name == 'rf':
        clf = RandomForestClassifier(**params)
    elif name == 'xgb':
        clf = XGBClassifier(**params)
    elif name == 'lgbm':
        clf = LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {name}")
    return clf
