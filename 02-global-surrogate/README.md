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

`surrogate.py`:

1. **Fidelity against size.** A tree with 8 leaves has R² = 0.56 and makes a different decision
   than `f` for 17% of the people. With 256 leaves R² is 0.85, but that tree is not readable.
2. The surrogate with 8 leaves: a short description with `age_at_first_arrest`, `employed`,
   `priors_count`, `male`, and `age`.
3. **What is missing.** For people with a drug charge, `f` gives a much lower risk to those
   in treatment (0.29 against 0.47). The surrogate does not use this feature at all (0.39
   against 0.40).
4. **Not truthful.** A second model is trained *without* `age_at_first_arrest`. Its surrogate
   still splits on `age_at_first_arrest` first: the surrogate found a correlation in the data,
   not the mechanism of the model.
5. **Random probe inputs** (when you do not have the training data): the surrogate fits the
   probes (R² 0.59) but not real people (R² 0.02), because random inputs are mostly impossible
   people.

## Tools

- [scikit-learn](https://scikit-learn.org): the standard machine-learning library for Python.
  Here: the opaque model (`HistGradientBoostingClassifier`), the surrogate
  (`DecisionTreeRegressor`), `export_text`, and `r2_score`.

## Run

With [uv](https://docs.astral.sh/uv/):

```sh
uv run surrogate.py
```
