import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def engineer_features(X, y, transforms):
    X = X.copy()
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
        col_names = [f"feat_{i}" for i in range(X.shape[1])]
        X.columns = col_names[: X.shape[1]]
    else:
        col_names = X.columns.tolist()

    if "target_encoding" in transforms:
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category").cat.codes
        for col in X.select_dtypes(include=["int", "float"]).columns[:5]:
            if X[col].nunique() > 10 and X[col].nunique() < 500:
                skf = KFold(n_splits=5, shuffle=True, random_state=42)
                te = np.zeros(len(X))
                for tr, va in skf.split(X):
                    te[va] = X.iloc[tr][col].map(
                        pd.Series(y[tr], index=X.iloc[tr][col]).groupby(X.iloc[tr][col]).mean()
                    ).fillna(y.mean()).values
                X[f"te_{col}"] = te

    if "interactions" in transforms:
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]
        for i in range(min(5, len(numeric_cols))):
            for j in range(i + 1, min(5, len(numeric_cols))):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                X[f"{c1}_x_{c2}"] = X[c1] * X[c2]

    return X.values
