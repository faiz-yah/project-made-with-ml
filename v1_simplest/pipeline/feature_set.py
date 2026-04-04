import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from pipeline.encoder import get_encoder
from pipeline.binners import get_binner

# Columns per feature set — mirrors x_basic_cols / x_one_cols from your EDA notebook
FEATURE_SETS = {
    "basic":    ["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
    "extended": ["Pclass", "Sex", "SibSp", "Parch", "Embarked",
                "Cabin_pattern", "Ticket_pattern",
                "Age_bin", "Fare_bin"],
}


def build_X(df_raw: pd.DataFrame, binner_name, encoder, feature_set: str, scaler: str, custom_bins=None):
    df = df_raw.copy()
    y = df["Survived"]

    # 1. Extract patterns from high-cardinality columns
    df = _obtain_pattern_ticket(df, "Ticket")
    df = _obtain_pattern_cabin(df, "Cabin")

    # 2. Bin continuous variables (Age and Fare)
    #    binner.fit_transform expects a Series; returns array or Series
    df["Age_bin"]  = get_binner(binner_name, col='Age', custom_bins=custom_bins).fit_transform(df[["Age"]].fillna(df["Age"].median()))
    df["Fare_bin"] = get_binner(binner_name, col='Fare', custom_bins=custom_bins).fit_transform(df[["Fare"]].fillna(df["Fare"].median()))

    # 3. Encode low-cardinality categoricals (Sex, Embarked)
    for col in ["Sex", "Embarked"]:
        df[col] = encoder.fit_transform(df[[col]].fillna("Missing"), y)

    # 4. Encode extracted patterns (Cabin_pattern, Ticket_pattern)
    if feature_set == "extended":
        for col in ["Cabin_pattern", "Ticket_pattern"]:
            df[col] = encoder.fit_transform(df[[col]].fillna("Missing"), y)

    # 5. Select columns for this feature set
    cols = FEATURE_SETS[feature_set]
    X = df[cols].copy()
    y = df["Survived"]

    # 6. Apply scaler
    if scaler == "standard":
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=cols)
    elif scaler == "minmax":
        X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=cols)
    # scaler == "none" → no transformation

    return X, y


def _obtain_pattern_ticket(df: pd.DataFrame, col: str) -> pd.DataFrame:
    first_char = df[col].str[0]
    df[f"{col}_pattern"] = np.where(
        first_char.str.isdigit(),
        "start_with_digit",
        "start_with_" + first_char.str.lower()
    )
    return df


def _obtain_pattern_cabin(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_pattern"] = df[col].str[0] 
    return df