"""Limits: small trees are unstable, and not each linear model or tree is interpretable."""

import numpy as np
from recidivism_data import load_xy, onehot
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

X_train, y_train = load_xy("train")
X_test, y_test = load_xy("test")
A_train, A_test = onehot(X_train), onehot(X_test)
names = list(A_train.columns)

print("1. The same tree learner on 6 random samples (with replacement) of the training data")
rng = np.random.default_rng(0)
predictions = []
for i in range(6):
    rows = rng.integers(0, len(A_train), len(A_train))
    tree = DecisionTreeClassifier(max_leaf_nodes=6, min_samples_leaf=50, random_state=0)
    tree.fit(A_train.iloc[rows], y_train.iloc[rows])
    t = tree.tree_
    splits = [
        f"{names[t.feature[n]]} <= {t.threshold[n]:.1f}"
        for n in range(t.node_count)
        if t.feature[n] >= 0
    ]
    predictions.append(tree.predict(A_test))
    print(f"   sample {i + 1}: accuracy {tree.score(A_test, y_test):.3f}, root: {splits[0]}")
    print(f"             then: {'; '.join(splits[1:])}")
agree = [np.mean(a == b) for i, a in enumerate(predictions) for b in predictions[i + 1 :]]
print(f"   two of these trees give the same prediction for {np.mean(agree):.0%} of the test people")
print(f"   (minimum {min(agree):.0%}): the 'explanation' depends on the sample")

print("\n2. A linear model is not interpretable only because it is linear")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
big = make_pipeline(poly, StandardScaler(), LogisticRegression(max_iter=2000)).fit(A_train, y_train)
terms = poly.get_feature_names_out(names)
weights = big[-1].coef_[0]
print(f"   logistic regression with all pairs of features: {len(terms)} terms,", end=" ")
print(f"accuracy {big.score(A_test, y_test):.3f}")
order = np.argsort(-np.abs(weights))[:8]
print("   the 8 largest of them: " + " ".join(f"{weights[i]:+.2f}*[{terms[i]}]" for i in order))
print("   which factor decides? what is the effect of age? You cannot read it from this.")

print("\n3. A tree is not interpretable only because it is a tree")
for leaves in [6, 50, 500, None]:
    tree = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(A_train, y_train)
    print(
        f"   {tree.get_n_leaves():5d} leaves, depth {tree.get_depth():2d}: accuracy {tree.score(A_test, y_test):.3f}"
    )
