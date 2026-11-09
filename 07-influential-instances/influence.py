"""Influential instances: which training rows change the model most?"""

import time
import warnings

import numpy as np
import pandas as pd
from recidivism_data import FEATURES, TARGET, load, load_xy, onehot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
C = 1.0

dirty = load("defendants_dirty").query("split == 'train'")
log = load("error_log")
kind = log.sort_values("column").groupby("id")["kind"].first()  # the value error, if there is one
clean_rows = dirty[~dirty["id"].isin(kind.index)].sample(1440, random_state=0)
sample = pd.concat([dirty[dirty["id"].isin(kind.index)], clean_rows]).sample(frac=1, random_state=0)
sample = sample.reset_index(drop=True)
is_error = sample["id"].isin(kind.index).to_numpy()
clean = load().set_index("id").loc[sample["id"]].reset_index()

A = onehot(sample[FEATURES]).to_numpy()
y = sample[TARGET].to_numpy()
X_valid, y_valid = load_xy("valid")
A_valid = onehot(X_valid).to_numpy()
names = list(onehot(sample[FEATURES]).columns)


def fit(A, y, start=None):
    model = LogisticRegression(C=C, max_iter=3000, tol=1e-8, warm_start=start is not None)
    if start is not None:
        model.coef_, model.intercept_ = start.coef_.copy(), start.intercept_.copy()
    return model.fit(A, y)


def theta(model) -> np.ndarray:
    return np.append(model.coef_[0], model.intercept_[0])


def quality(model) -> tuple[float, float]:
    return model.score(A_valid, y_valid), log_loss(y_valid, model.predict_proba(A_valid)[:, 1])


print("1. A model on 1500 training rows; 60 of them have a data-entry error")
f = fit(A, y)
f_clean = fit(onehot(clean[FEATURES]).to_numpy(), clean[TARGET].to_numpy())
for label, model in [("with the errors", f), ("without errors", f_clean)]:
    accuracy, loss = quality(model)
    weights = dict(zip(names, model.coef_[0]))
    print(f"   {label}: accuracy {accuracy:.3f}, log loss {loss:.4f},", end=" ")
    print(f"weight of priors_count {weights['priors_count']:+.3f}, of age {weights['age']:+.3f}")
print("   Is the model skewed by a few rows? Which ones?")

print("\n2. Leave one out: train the model again without row i, for each i")
start = time.perf_counter()
base_loss = quality(f)[1]
change_theta = np.zeros((len(y), len(theta(f))))
change_loss = np.zeros(len(y))
for i in range(len(y)):
    keep = np.arange(len(y)) != i
    g = fit(A[keep], y[keep], start=f)
    change_theta[i] = theta(g) - theta(f)
    change_loss[i] = quality(g)[1] - base_loss
seconds_loo = time.perf_counter() - start
scale = np.append(A.std(axis=0), 1.0)
influence = np.linalg.norm(change_theta * scale, axis=1)  # change of the parameters
print(f"   {len(y)} models trained in {seconds_loo:.0f} s")


def report(score: np.ndarray, label: str) -> None:
    order = np.argsort(-score)
    hits = {k: int(is_error[order[:k]].sum()) for k in (20, 60)}
    print(f"   {label}: {hits[20]} of the top 20 and {hits[60]} of the top 60 are injected errors")
    kinds = kind.reindex(sample.loc[order[:60], "id"]).value_counts().to_dict()
    print(f"      kinds in the top 60: {kinds}")


print("\n3. The most influential rows, compared with the log of the injected errors")
report(influence, "largest change of the parameters")
report(-change_loss, "largest improvement on validation data when removed")
top = np.argsort(-influence)[:5]
columns = ["id", "age", "priors_count", TARGET]
print("   the five rows with the largest change of the parameters (none is an error):")
print(
    "   " + sample.loc[top, columns].assign(error=is_error[top]).to_string().replace("\n", "\n   ")
)
found = set(np.argsort(change_loss)[:60])
by_kind = pd.Series(kind.reindex(sample["id"]).to_numpy())[is_error]
share = by_kind.groupby(by_kind).apply(lambda s: np.mean([i in found for i in s.index]))
print(
    "   share of each kind of error in the top 60 (validation): "
    + ", ".join(f"{k} {v:.0%}" for k, v in share.items())
)

print("\n4. Without retraining: influence functions (one Hessian, one gradient per row)")
start = time.perf_counter()
A1 = np.column_stack([A, np.ones(len(y))])
p = f.predict_proba(A)[:, 1]
gradients = A1 * (p - y)[:, None]
regularization = np.diag(np.append(np.ones(A.shape[1]), 0.0)) / C
hessian = (A1 * (p * (1 - p))[:, None]).T @ A1 + regularization
approx_theta = np.linalg.solve(hessian, gradients.T).T  # removing row i changes theta by this
seconds_if = time.perf_counter() - start
approx = np.linalg.norm(approx_theta * scale, axis=1)
print(f"   computed in {seconds_if * 1000:.0f} ms ({seconds_loo / seconds_if:.0f} times faster)")
print(f"   correlation with the leave-one-out result: {np.corrcoef(approx, influence)[0, 1]:.3f}")
report(approx, "largest approximate change of the parameters")

print("\n5. Use: remove the rows whose removal helps most on validation data")
order = np.argsort(change_loss)
for k in [0, 20, 60]:
    keep = np.ones(len(y), bool)
    keep[order[:k]] = False
    accuracy, loss = quality(fit(A[keep], y[keep]))
    print(f"   without the top {k:2d}: accuracy {accuracy:.3f}, log loss {loss:.4f}")
accuracy, loss = quality(f_clean)
print(f"   all errors corrected: accuracy {accuracy:.3f}, log loss {loss:.4f}")

print("\n6. Limit: one row at a time does not show the effect of a group")
priors = names.index("priors_count")
group = np.flatnonzero(kind.reindex(sample["id"]).to_numpy() == "priors_plus_90")
single = change_theta[group, priors]
together = (
    fit(np.delete(A, group, axis=0), np.delete(y, group)).coef_[0][priors] - f.coef_[0][priors]
)
print(f"   weight of priors_count: {f.coef_[0][priors]:+.3f}")
print(
    f"   remove one of the {len(group)} rows with priors + 90: at most {np.abs(single).max():+.3f}"
)
print(f"   sum of the {len(group)} single effects: {single.sum():+.3f}")
print(f"   remove all {len(group)} together: {together:+.3f}")
