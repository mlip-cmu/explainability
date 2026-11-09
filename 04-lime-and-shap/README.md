# 04 · Explaining one prediction: LIME and SHAP

A court uses an opaque gradient boosting model that predicts whether a defendant is rearrested
within two years ([dataset and model](../recidivism-data/)). The model gives a 24-year-old man
with five priors and a drug charge a risk of 0.69.

**Problem.** The defendant and the judge want to know why: which features were most
influential for *this* prediction? Feature importance (project 03) answers for the whole model,
not for one person.

**Idea.** Two ways to give each feature a share of one prediction.
[LIME](https://arxiv.org/abs/1602.04938) fits a simple linear model to the predictions of the
opaque model for inputs near the person (a *local surrogate*).
[SHAP](https://arxiv.org/abs/1705.07874) divides the difference between this prediction and the
average prediction among the features with Shapley values from game theory. Both describe the
model, not the world.

LIME in four steps, by hand, for a model with two features (`lime_and_shap.ipynb`):

```python
samples = X_train[two].sample(600, random_state=0).to_numpy(float)  # step 1
predictions = gbm2.predict_proba(samples)[:, 1]  # step 2
distance = np.sqrt((((samples - center) / scale) ** 2).sum(axis=1))
weights = np.exp(-(distance**2) / 0.75**2)  # step 3
local = Ridge(alpha=1.0).fit(samples - center, predictions, sample_weight=weights)  # step 4
```

SHAP values for a tree model are exact, and they add up to the prediction:

```python
shap_explainer = shap.TreeExplainer(gbm)
shap_one = shap_explainer(one_encoded)[0]
total = shap_one.base_values + shap_one.values.sum()
```

## What the code shows

Open [`lime_and_shap.ipynb`](lime_and_shap.ipynb) on GitHub to see the outputs.

- `lime_and_shap.ipynb`:
  1. LIME by hand: the surface of the opaque model, the weighted samples, and the local plane
     (−0.016 per year of age, +0.044 per prior). The plane is right near the person and wrong
     far away.
  2. LIME with the library: a short explanation with ranges (`priors_count > 3`,
     `age <= 25`, `employed=0`). The local model has R² = 0.65 and predicts 0.73 where the
     opaque model says 0.69.
  3. SHAP waterfall and force plot: average log-odds −0.49 plus the SHAP values (+1.29) is the
     log-odds of the model (+0.80, P = 0.69). All features get a value.
  4. **SHAP against the truth.** The model gives +0.27 to `age_at_first_arrest`, which has no
     effect in the true process, and too little to `priors_count` (+0.33, truth +0.65) and
     `age` (+0.15, truth +0.43). SHAP describes the model faithfully; it does not describe
     the world.
  5. SHAP for 1 500 people: beeswarm and dependence plots (explanations of single predictions
     put together to explain the model), and the ranking for two people against the global one.
- `lime_stability.py`: the same prediction explained 20 times.
  - With the default 5 000 samples the same five features are always there; the values change
    by about 10% (e.g. `age <= 25`: 0.112 to 0.129) and two different top-3 orders occur.
  - With 500 samples there are six different top-3 lists, and two features are only sometimes
    among the five.
  - A small neighborhood (kernel width 0.5) gives much smaller values (+0.06 in place of +0.15
    for the priors).
  - `TreeExplainer` gives exactly the same SHAP values in each run.

## Tools

- [LIME](https://github.com/marcotcr/lime): local surrogate explanations for any classifier.
  Here: `LimeTabularExplainer` for the tabular data. The package has had no release since 2020,
  but works with current NumPy and scikit-learn.
- [SHAP](https://shap.readthedocs.io): Shapley-value explanations with fast algorithms for
  trees and other model types. Here: `TreeExplainer` and the waterfall, force, beeswarm, and
  scatter plots.
- [scikit-learn](https://scikit-learn.org): the opaque model and the `Ridge` regression of the
  manual LIME.
- [Matplotlib](https://matplotlib.org) and [Jupyter](https://jupyter.org): charts and notebook.

## Run

With [uv](https://docs.astral.sh/uv/):

```sh
uv run lime_stability.py
uv run jupyter nbconvert --to notebook --execute --inplace lime_and_shap.ipynb
```
