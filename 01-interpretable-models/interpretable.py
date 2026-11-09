"""Inherently interpretable models: the model is the explanation."""

import warnings

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from recidivism_data import load, load_xy, onehot, slide_examples
from recidivism_data.shared import opaque_model
from rule_list import learn_rule_list, predict_rule_list
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")

X_train, y_train = load_xy("train")
X_test, y_test = load_xy("test")
A_train, A_test = onehot(X_train), onehot(X_test)
results = {}


def record(name, predictions, size):
    results[name] = (np.mean(predictions == y_test), size)


def conditions(X: pd.DataFrame) -> pd.DataFrame:
    """Yes/no statements about a defendant: the vocabulary of score cards and rule lists."""
    return pd.DataFrame(
        {
            "age 18-20": X["age"] <= 20,
            "age 21-23": X["age"].between(21, 23),
            "age 24-29": X["age"].between(24, 29),
            "age 45+": X["age"] >= 45,
            "male": X["sex"] == "male",
            "no priors": X["priors_count"] == 0,
            "priors 2-3": X["priors_count"].between(2, 3),
            "priors >3": X["priors_count"] > 3,
            "juvenile priors": X["juvenile_priors"] > 0,
            "employed": X["employed"] == 1,
            "drug charge": X["charge_type"] == "drug",
            "traffic charge": X["charge_type"] == "traffic",
            "drug charge, in treatment": (X["charge_type"] == "drug") & (X["in_treatment"] == 1),
            "felony": X["charge_degree"] == "felony",
            "county south": X["county"] == "south",
        }
    ).astype(int)


C_train, C_test = conditions(X_train), conditions(X_test)

print("1. Sparse linear model (logistic regression with L1 penalty)")
print("   a stronger penalty removes features; accuracy on the test data:")
for c in [1.0, 0.01, 0.003, 0.0015]:
    lasso = LogisticRegression(penalty="l1", C=c, solver="liblinear")
    sparse = make_pipeline(StandardScaler(), lasso).fit(A_train, y_train)
    used = int((lasso.coef_[0] != 0).sum())
    print(
        f"   C={c:<6} {used:2d} of {A_train.shape[1]} features, accuracy {sparse.score(A_test, y_test):.3f}"
    )
    if c == 1.0:
        record("logistic regression, all features", sparse.predict(A_test), f"{used} terms")
    if c == 0.01:
        record("sparse logistic regression", sparse.predict(A_test), f"{used} terms")
        kept = [
            (n, w / s) for n, w, s in zip(A_train.columns, lasso.coef_[0], sparse[0].scale_) if w
        ]
        offset = lasso.intercept_[0] - sum(
            w * m for (_, w), m in zip(kept, sparse[0].mean_[lasso.coef_[0] != 0])
        )
        terms = " ".join(f"{w:+.3f}*{n}" for n, w in kept)
        print(f"   the complete model for C=0.01:\n   P(rearrest) = sigmoid({offset:+.2f} {terms})")

print("\n2. Score card: a sparse linear model on yes/no conditions, with round points")
lasso = LogisticRegression(penalty="l1", C=0.02, solver="liblinear").fit(C_train, y_train)
points = pd.Series(np.round(lasso.coef_[0] / 0.35).astype(int), index=C_train.columns)
points = points[points != 0]
for name, value in points.items():
    print(f"   {name:<28}{value:+d}")
score_train, score_test = C_train[points.index] @ points, C_test[points.index] @ points
risk = y_train.groupby(score_train.clip(-2, 5)).mean()
print("   total points:  " + "  ".join(f"{s:>4d}" for s in risk.index) + "   (clipped to -2..5)")
print("   rearrest rate: " + "  ".join(f"{r:>4.0%}" for r in risk))
threshold = min(s for s in risk.index if risk[s] > 0.5)
print(f"   predict a rearrest from {threshold} point(s)")
record("score card", (score_test >= threshold).astype(int), f"{len(points)} conditions")

print("\n3. Small decision tree")
tree = DecisionTreeClassifier(max_leaf_nodes=6, min_samples_leaf=50, random_state=0).fit(
    A_train, y_train
)
print("   " + export_text(tree, feature_names=list(A_train.columns)).replace("\n", "\n   "))
record("decision tree, 6 leaves", tree.predict(A_test), f"{tree.get_n_leaves()} leaves")

print("4. Rule lists")


def lecture_rule_list(X):
    return (
        (X["age"].between(18, 20) & (X["sex"] == "male"))
        | (X["age"].between(21, 23) & X["priors_count"].between(2, 3))
        | (X["priors_count"] > 3)
    ).astype(int)


print("   a) a rule list from the literature (learned with CORELS on real data):")
print("      IF age 18-20 and male THEN arrest, ELSE IF age 21-23 and 2-3 priors THEN arrest,")
print("      ELSE IF more than 3 priors THEN arrest, ELSE no arrest")
record("rule list from the literature", lecture_rule_list(X_test), "3 rules")
rules = learn_rule_list(C_train, y_train)
print("   b) a rule list learned from our training data (rule_list.py):")
for i, (rule, n, rate) in enumerate(rules):
    start = "IF" if i == 0 else "ELSE IF"
    print(f"      {start} {' and '.join(rule)} THEN arrest   ({n} people, {rate:.0%} rearrested)")
print("      ELSE no arrest")
record("learned rule list", predict_rule_list(rules, C_test), f"{len(rules)} rules")

print("\n5. A prediction of an interpretable model explains itself")
example = slide_examples().iloc[[0]]
print("   18-year-old male, 1 prior:")
print("   rule list: the first rule applies (age 18-20 and male), thus: predict arrest.")
print("   counterfactual from the same rule list: no arrest if he were 21 or older, or female.")
active = [n for n in points.index if conditions(example)[n].iloc[0]]
total = int(conditions(example)[points.index].iloc[0] @ points)
print(
    "   score card: " + ", ".join(f"{n} ({points[n]:+d})" for n in active) + f" = {total} point(s)"
)

print("\n6. Accuracy on the test data: is there a price for interpretability?")
ebm = ExplainableBoostingClassifier(interactions=0, random_state=0).fit(X_train, y_train)
record("explainable boosting machine (GAM)", ebm.predict(X_test), "1 curve per feature")
record("gradient boosting (opaque)", opaque_model().predict(X_test), "150 trees")
truth = load().query("split == 'test'")["true_probability"].to_numpy()
record("the true process (upper limit)", (truth > 0.5).astype(int), "-")
record("always predict no rearrest", np.zeros(len(y_test), int), "-")
for name, (accuracy, size) in sorted(results.items(), key=lambda r: -r[1][0]):
    print(f"   {name:<38}{accuracy:.3f}   {size}")
