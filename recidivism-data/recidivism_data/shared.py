"""The opaque model and the person that several projects explain."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from . import FEATURES, load, load_xy, onehot

PERSON_ID = 1657


def opaque_model(table: str = "defendants") -> Pipeline:
    """Gradient boosting on the encoded features. `model[-1]` is the boosting model itself."""
    X, y = load_xy("train", table)
    gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, random_state=0)
    return make_pipeline(FunctionTransformer(onehot), gbm).fit(X, y)


def person(person_id: int | None = None) -> pd.DataFrame:
    """One defendant from the test split (one row, the feature columns)."""
    df = load()
    return df.loc[df["id"] == (person_id or PERSON_ID), FEATURES]
