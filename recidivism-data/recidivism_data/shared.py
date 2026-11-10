"""The opaque model and the person that several projects explain."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from . import FEATURES, load, load_xy, onehot

PERSON_ID = 1657


def new_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40, random_state=0
    )


def opaque_model(table: str = "defendants") -> Pipeline:
    """Gradient boosting on the encoded features. `model[-1]` is the boosting model itself."""
    X, y = load_xy("train", table)
    return make_pipeline(FunctionTransformer(onehot), new_gbm()).fit(X, y)


def person(person_id: int | None = None) -> pd.DataFrame:
    """One defendant from the test split (one row, the feature columns)."""
    df = load()
    return df.loc[df["id"] == (person_id or PERSON_ID), FEATURES]


def age_model():
    """The same boosted trees with age as the only feature: a model that fits in a 2D plot.

    Returns the model and a function that gives the predicted risk for a list of ages.
    """
    X, y = load_xy("train")
    model = new_gbm().fit(X[["age"]].astype(float), y)

    def risk(ages) -> np.ndarray:
        return model.predict_proba(pd.DataFrame({"age": np.asarray(ages, float)}))[:, 1]

    return model, risk
