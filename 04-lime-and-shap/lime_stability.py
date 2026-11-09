"""How stable is a LIME explanation? Explain the same prediction again and again."""

import warnings

import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from recidivism_data import load_xy, onehot
from recidivism_data.shared import opaque_model, person

warnings.filterwarnings("ignore")

X_train, _ = load_xy("train")
A_train = onehot(X_train)
names = list(A_train.columns)
binary = [i for i, n in enumerate(names) if A_train[n].nunique() == 2]
gbm = opaque_model()[-1]
one = onehot(person()).to_numpy()[0]


def explain(seed: int, num_samples: int = 5000, kernel_width: float | None = None) -> pd.Series:
    explainer = lime.lime_tabular.LimeTabularExplainer(
        A_train.to_numpy(),
        feature_names=names,
        categorical_features=binary,
        kernel_width=kernel_width,
        random_state=seed,
    )
    e = explainer.explain_instance(one, gbm.predict_proba, num_features=5, num_samples=num_samples)
    return pd.Series(dict(e.as_list()))


def report(runs: list[pd.Series]) -> None:
    table = pd.DataFrame(runs)
    top3 = {tuple(r.abs().sort_values(ascending=False).index[:3]) for r in runs}
    summary = pd.DataFrame(
        {
            "in top 5": table.notna().sum().astype(str) + f" of {len(runs)}",
            "min": table.min(),
            "max": table.max(),
        }
    ).sort_values("max", ascending=False)
    print("   " + summary.round(3).to_string().replace("\n", "\n   "))
    print(f"   different top-3 lists (in order): {len(top3)}")


print("1. LIME, 20 runs with the default 5000 samples: the 5 most influential features")
report([explain(seed) for seed in range(20)])

print("\n2. LIME, 20 runs with 500 samples (faster)")
report([explain(seed, num_samples=500) for seed in range(20)])

print("\n3. LIME with a different size of the neighborhood (kernel width), same seed")
for width in [0.5, 1.0, None, 10.0]:
    e = explain(0, kernel_width=width)
    label = "default (2.9)" if width is None else width
    print(f"   width {label!s:<14}" + ", ".join(f"{k} {v:+.2f}" for k, v in e.items()))

print("\n4. SHAP for trees is exact: 20 runs")
values = [shap.TreeExplainer(gbm)(one.reshape(1, -1)).values[0] for _ in range(20)]
print(f"   largest difference between two runs: {np.ptp(values, axis=0).max():.6f}")
