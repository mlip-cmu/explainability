"""Global surrogate: explain an opaque model with an interpretable model of its predictions."""

import numpy as np
import pandas as pd
from recidivism_data import load_xy, onehot
from recidivism_data.shared import new_gbm, opaque_model
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text

X_train, y_train = load_xy("train")
X_test, y_test = load_xy("test")
A_train, A_test = onehot(X_train), onehot(X_test)
names = list(A_train.columns)

model = opaque_model()
f_train = model.predict_proba(X_train)[:, 1]
f_test = model.predict_proba(X_test)[:, 1]


def fidelity(surrogate, A=A_test, target=f_test):
    g = surrogate.predict(A)
    return r2_score(target, g), np.mean((g > 0.5) == (target > 0.5))


print("1. Train a tree g on the predictions of the opaque model f (not on the labels)")
print("   leaves  R2(f, g)  same decision as f   accuracy of g on the labels")
for leaves in [2, 4, 8, 16, 64, 256]:
    g = DecisionTreeRegressor(max_leaf_nodes=leaves, random_state=0).fit(A_train, f_train)
    r2, same = fidelity(g)
    accuracy = np.mean((g.predict(A_test) > 0.5) == y_test)
    print(f"   {leaves:5d}   {r2:6.2f}   {same:12.0%}   {accuracy:17.3f}")
print(f"   (accuracy of f on the labels: {model.score(X_test, y_test):.3f})")

print("\n2. The surrogate with 8 leaves, as an explanation of f")
g8 = DecisionTreeRegressor(max_leaf_nodes=8, random_state=0).fit(A_train, f_train)
print("   " + export_text(g8, feature_names=names, decimals=2).replace("\n", "\n   "))
r2, same = fidelity(g8)
print(f"It explains {r2:.0%} of the variance of f; {1 - same:.0%} of the decisions are different.")

print("\n3. What the surrogate does not show")
drug = X_test["charge_type"] == "drug"
table = pd.DataFrame({"f": f_test, "g": g8.predict(A_test)})[drug].groupby(
    X_test.loc[drug, "in_treatment"].map({0: "not in treatment", 1: "in treatment"})
)
print("   mean predicted risk of people with a drug charge:")
print("   " + table.mean().round(2).to_string().replace("\n", "\n   "))
used = sorted({names[i] for i in g8.tree_.feature if i >= 0})
print(f"   the surrogate uses only: {', '.join(used)}")

print("\n4. The surrogate can use a feature that the model cannot see")
blind = [n for n in names if n != "age_at_first_arrest"]
f2 = new_gbm().fit(A_train[blind], y_train)
g2 = DecisionTreeRegressor(max_leaf_nodes=8, random_state=0)
g2.fit(A_train, f2.predict_proba(A_train[blind])[:, 1])
r2, _ = fidelity(g2, target=f2.predict_proba(A_test[blind])[:, 1])
root = names[g2.tree_.feature[0]]
print("   a second model f2 is trained without age_at_first_arrest")
print(f"   its surrogate (R2 {r2:.2f}) has this first split: {root} <= {g2.tree_.threshold[0]:.1f}")
print("   the surrogate describes a correlation in the data, not the mechanism of the model")

print("\n5. Without access to the training data: random probe inputs")
rng = np.random.default_rng(0)
probe = pd.DataFrame({n: rng.choice(A_train[n].unique(), 5000) for n in names})
probe_target = model[-1].predict_proba(probe)[:, 1]
g3 = DecisionTreeRegressor(max_leaf_nodes=8, random_state=0).fit(probe, probe_target)
r2_probe, _ = fidelity(g3, probe, probe_target)
r2_real, same = fidelity(g3)
print(
    f"   R2 on the probe inputs {r2_probe:.2f}, on real people {r2_real:.2f} ({same:.0%} same decision)"
)
print(f"   first split: {names[g3.tree_.feature[0]]} <= {g3.tree_.threshold[0]:.1f}")
print("   random inputs are mostly impossible people (age 19 with 30 priors);")
print("   the surrogate explains the model where no real person is")
