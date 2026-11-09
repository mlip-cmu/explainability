"""Opaque models: you can print every parameter, and still not understand the model."""

import warnings

import numpy as np
from recidivism_data import load_xy, onehot
from recidivism_data.shared import opaque_model, person
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
X_train, y_train = load_xy("train")
X_test, y_test = load_xy("test")

print("1. Gradient boosting")
model = opaque_model()
gbm = model[-1]
trees = [t[0] for t in gbm._predictors]
nodes = sum(len(t.nodes) for t in trees)
print(f"   {len(trees)} trees, {nodes} nodes, test accuracy {model.score(X_test, y_test):.3f}")
print("   the first 6 of the", len(trees[0].nodes), "nodes of tree 1:")
names = list(onehot(X_train).columns)
for node in trees[0].nodes[:6]:
    if node["is_leaf"]:
        print(f"     leaf: add {node['value']:+.4f} to the log-odds")
    else:
        print(f"     if {names[node['feature_idx']]} <= {node['num_threshold']:.1f} ...")
print("   a prediction is the sum of one leaf of each of the", len(trees), "trees")

print("\n2. Neural network")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), early_stopping=True, random_state=0)
net = make_pipeline(StandardScaler(), mlp).fit(onehot(X_train), y_train)
params = sum(w.size + b.size for w, b in zip(mlp.coefs_, mlp.intercepts_))
print(f"   layers {mlp.hidden_layer_sizes}, {params} parameters,", end=" ")
print(f"test accuracy {net.score(onehot(X_test), y_test):.3f}")
print("   the first 4 x 6 of the", mlp.coefs_[0].size, "weights of layer 1:")
print("   " + np.array2string(mlp.coefs_[0][:4, :6], precision=3, prefix="   "))

print("\n3. The question that these parameters do not answer")
p = person()
print("  ", p.iloc[0].to_dict())
print(f"   gradient boosting: P(rearrest) = {model.predict_proba(p)[0, 1]:.2f}")
print(f"   neural network:    P(rearrest) = {net.predict_proba(onehot(p))[0, 1]:.2f}")
print("   Why? Which factors? What must change for a different result?")
