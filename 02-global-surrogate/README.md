# 02 · Global surrogate models

A court uses an opaque gradient boosting model (150 trees) that predicts whether a defendant is
rearrested within two years ([dataset and model](../recidivism-data/)).

**Problem.** The court wants to know how the model decides in general, but nobody can read 150
trees.

**Idea.** Ask the opaque model `f` for its predictions on a data set, and train an
interpretable model `g` (here: a small tree) on those predictions. Then read `g`. How well `g`
describes `f` is measurable with normal quality measures, typically R². The method needs no
access to the internals of `f`. But `g` is a second model: what it says is not necessarily what
`f` does.

The surrogate learns the predictions of the model, not the labels (`surrogate.py`):

```python
f_train = model.predict_proba(X_train)[:, 1]
g = DecisionTreeRegressor(max_leaf_nodes=leaves, random_state=0).fit(A_train, f_train)
r2_score(f_test, g.predict(A_test))
```

## What the code shows

[`illustration.ipynb`](illustration.ipynb) shows the idea in pictures. For a model with only
the age as a feature, the model is one curve; a linear regression on 8 predictions of the model
is the surrogate (risk = 0.70 − 0.009 · age, R² = 0.72). For a model with age and priors, a
tree with 6 leaves replaces the surface of the model with 6 rectangles (R² = 0.78).

`surrogate.py`, for the model with all features:

1. **Fidelity against size.** A tree with 8 leaves has R² = 0.70 and makes a different decision
   than `f` for 16% of the people. With 256 leaves R² is 0.95, but that tree is not readable.
2. The surrogate with 8 leaves: a short description with `age_at_first_arrest`, `employed`,
   `priors_count`, and `age`.
3. **What is missing.** For people with a drug charge, `f` gives a much lower risk to those
   in treatment (0.30 against 0.45). The surrogate does not use this feature at all (0.39
   against 0.39).
4. **Not truthful.** A second model is trained *without* `age_at_first_arrest`. Its surrogate
   still splits on `age_at_first_arrest` first: the surrogate found a correlation in the data,
   not the mechanism of the model.
5. **Random probe inputs** (when you do not have the training data): the surrogate fits real
   people less well (R² 0.45 in place of 0.70, a different decision for 23% of them) and has a
   different first split (`age`), because random inputs are mostly impossible people.

## Tools

- [scikit-learn](https://scikit-learn.org): the standard machine-learning library for Python.
  Here: the opaque model (`HistGradientBoostingClassifier`), the surrogate
  (`DecisionTreeRegressor`), `export_text`, and `r2_score`.
- [Matplotlib](https://matplotlib.org) and [Jupyter](https://jupyter.org): the charts and the
  notebook.

## Run

With [uv](https://docs.astral.sh/uv/):

```sh
uv run surrogate.py
uv run jupyter nbconvert --to notebook --execute --inplace illustration.ipynb
```
