"""The same comparison on real data: the COMPAS records that ProPublica published."""

import sys
from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

try:
    raw = pd.read_csv(StringIO(requests.get(URL, timeout=30).text))
except (requests.RequestException, pd.errors.ParserError) as error:
    print(f"no download possible ({type(error).__name__}); this script needs the network")
    sys.exit(0)

days = (pd.to_datetime(raw["c_jail_out"]) - pd.to_datetime(raw["c_jail_in"])).dt.days
X = pd.DataFrame(
    {
        "age": raw["age"],
        "male": (raw["sex"] == "Male").astype(int),
        "priors_count": raw["priors_count"],
        "juvenile_priors": raw[["juv_fel_count", "juv_misd_count", "juv_other_count"]].sum(axis=1),
        "felony": (raw["c_charge_degree"] == "F").astype(int),
        "days_in_jail": days.fillna(0).clip(0, 365),
    }
)
y = raw["two_year_recid"]
print(f"{len(X)} defendants from Broward County, Florida; {y.mean():.0%} rearrested in two years")

rule_list = (
    (X["age"].between(18, 20) & (X["male"] == 1))
    | (X["age"].between(21, 23) & X["priors_count"].between(2, 3))
    | (X["priors_count"] > 3)
).astype(int)
models = {
    "gradient boosting (opaque)": HistGradientBoostingClassifier(max_iter=100, random_state=0),
    "logistic regression, 6 features": make_pipeline(StandardScaler(), LogisticRegression()),
    "decision tree, 6 leaves": DecisionTreeClassifier(max_leaf_nodes=6, random_state=0),
    "always predict no rearrest": DummyClassifier(),
}
rows = [("rule list from the literature, 3 rules", np.mean(rule_list == y))]
for name, model in models.items():
    rows.append((name, np.mean(cross_val_predict(model, X, y, cv=10) == y)))
compas = raw["score_text"].isin(["Medium", "High"]).astype(int)
rows.append(("COMPAS score (medium or high)", np.mean(compas == y)))
print("accuracy (10-fold cross-validation for the learned models):")
for name, accuracy in sorted(rows, key=lambda r: -r[1]):
    print(f"   {name:<42}{accuracy:.3f}")
