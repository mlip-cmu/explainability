# 03 · Feature importance and partial dependence plots

A court uses an opaque gradient boosting model that predicts whether a defendant is rearrested
within two years ([dataset and model](../recidivism-data/)).

**Problem.** Two questions about the whole model, which its 150 trees do not answer directly:
on which features does the model rely, and how does the prediction change with a feature (more
priors, a higher age)?

**Idea.** Ask the model, do not read it. *Permutation feature importance:* shuffle one feature
in the test data, so that it has no information, and measure how much the accuracy decreases.
*Partial dependence plot:* set one feature to a value for all people, average the predictions,
and repeat for each value. Both need only predictions, no retraining and no access to the
internals.

Shuffle one feature by hand, then let scikit-learn do it for all features with repeats:

```python
shuffled = X_test.copy()
shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
accuracy = model.score(shuffled, y_test)

result = permutation_importance(estimator, X_test, y_test, n_repeats=30, random_state=0)
```

A partial dependence plot is the average of the curves of many people:

```python
ice = np.array([curve(sample.iloc[[i]], predict_model) for i in range(len(sample))])
ax.plot(ages, ice.mean(axis=0), label="average = partial dependence (model)")
```

## What the code shows

Open [`importance_and_pdp.ipynb`](importance_and_pdp.ipynb) on GitHub to see the outputs.

1. **Importance.** Shuffling `priors_count` decreases the accuracy from 0.675 to 0.639;
   shuffling `county` changes nothing. The most important features are `age` (0.031) and
   `priors_count` (0.027).
2. **Importance is about the model, not about the world.** The third feature of the model is
   `age_at_first_arrest` (0.017). In the true process, which the notebook measures in the same
   way, its importance is 0: it is only correlated with age and priors.
3. **Correlated features.** Shuffling `age` costs 0.031, but a model retrained without `age`
   loses only 0.014, because other features take its place. Without `age_at_first_arrest` the
   retrained model is even a little better (+0.005).
4. **Partial dependence.** The curve of one person, the curves of 60 people, and their average.
   The risk decreases quickly from age 18 to 30. The model's curve starts lower than the true
   curve (0.53 against 0.70 at age 18), because the model gives a part of the age effect to
   `age_at_first_arrest`.
5. **Interaction.** Treatment decreases the risk for a drug charge by 0.14 (truth: 0.18) and
   almost not for other charges. A plot for `in_treatment` alone shows only a small average
   effect.
6. **Impossible people.** For the point "age 20" of the curve, 50% of the people get a first
   arrest that is after their age.

## Tools

- [scikit-learn](https://scikit-learn.org): the standard machine-learning library for Python.
  Here: `permutation_importance` and `PartialDependenceDisplay` from `sklearn.inspection`.
- [Matplotlib](https://matplotlib.org): the standard plotting library. Here: all charts.
- [Jupyter](https://jupyter.org): notebooks with code, text, and outputs in one file.

## Run

To change the notebook and run it again, with [uv](https://docs.astral.sh/uv/):

```sh
uv run jupyter lab importance_and_pdp.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace importance_and_pdp.ipynb
```
