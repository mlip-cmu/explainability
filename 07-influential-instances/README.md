# 07 · Influential instances

A court uses a model that predicts whether a defendant is rearrested within two years
([dataset](../recidivism-data/)). This project uses 1 500 rows of the dirty training data:
1 440 correct rows and the 60 rows with a data-entry error (`priors_count + 90`, `age + 100`,
or a flipped label), with the log of the errors as ground truth.

**Problem.** A logistic regression trained on these rows is bad: 0.647 accuracy in place of
0.681, and the weight of `priors_count` is negative (more priors, less risk). Is the model
skewed by a few rows, and by which ones? Unusual feature values are easy to find, but which
rows changed *the model*?

**Idea.** A training row is influential if the model is different without it. *Leave one out:*
train the model again without row `i`, for each `i`, and measure the change of the parameters
or of the loss on validation data. For some models, e.g. logistic regression, an *influence
function* gives almost the same number without retraining, from one Hessian matrix and the
gradient of each row.

Leave one out (`influential_instances.ipynb`):

```python
for i in range(len(y)):
    keep = np.arange(len(y)) != i
    g = fit(A[keep], y[keep], start=f)
    change_theta[i] = theta(g) - theta(f)
    change_loss[i] = quality(g)[1] - base_loss
```

The same without retraining:

```python
gradients = A1 * (p - y)[:, None]
hessian = (A1 * (p * (1 - p))[:, None]).T @ A1 + regularization
approx_theta = np.linalg.solve(hessian, gradients.T).T  # removing row i changes theta by this
```

## What the code shows

Open [`influential_instances.ipynb`](influential_instances.ipynb) on GitHub to see the outputs.

1. **The idea in two pictures**, for models with only the age as a feature.
   - One added person (85 years old, rearrested) changes boosted trees: the risk at age 85 is
     0.96 with the person and 0.15 without. This works only because the person is alone in
     that region and a leaf may hold one person; with at least 40 people in a leaf the change
     is 0.13 to 0.15.
   - One wrong record (a 25-year-old entered as 125) changes a linear model from
     0.63 − 0.0070 · age to 0.45 − 0.0027 · age.
2. **The influential rows of a data set with errors**, for a model with all features.
   - The model with the 60 errors against the model on corrected data (see the problem above).
   - Leave one out: 1 500 models, about 40 seconds.
   - **Influential is not the same as wrong.** The rows that change the parameters most are
     correct but unusual, e.g. a 75-year-old who was rearrested, or a person with 18 priors
     who was not. Only 7 of the top 60 are injected errors. The rows whose removal *improves
     the validation loss* most are a better list for debugging: 10 of the top 20 are injected
     errors, and 70% of the `age + 100` rows are in the top 60.
   - **Influence functions** give the same ranking (correlation 1.000 with leave one out) in
     some milliseconds.
   - Without the 60 rows whose removal helps most, the accuracy increases from 0.647 to 0.668
     (all errors corrected: 0.681).
   - **Groups hide from this method.** None of the 20 rows with `priors_count + 90` is in the
     top 60. Without one of them, the weight of `priors_count` changes by at most 0.001,
     because the other 19 hold it in place; the 20 single effects add up to +0.011. Without
     all 20 together, the weight changes by +0.097 and gets its correct sign.

## Tools

- [scikit-learn](https://scikit-learn.org): logistic regression, with `warm_start` so that each
  leave-one-out model starts from the complete model.
- [NumPy](https://numpy.org): the influence function (gradients, Hessian, one linear solve).
- [Matplotlib](https://matplotlib.org) and [Jupyter](https://jupyter.org): the charts and the
  notebook.

## Run

To change the notebook and run it again, with [uv](https://docs.astral.sh/uv/) (about one
minute):

```sh
uv run jupyter lab influential_instances.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace influential_instances.ipynb
```
