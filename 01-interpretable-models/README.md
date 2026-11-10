# 01 · Inherently interpretable models

A court uses a model that predicts whether a defendant is rearrested within two years. The
data is the synthetic `defendants` table of the [dataset package](../recidivism-data/).

**Problem.** A gradient boosting model for this task has 150 trees with 2 162 nodes, a small
neural network has 3 649 weights. You can print all of them, and you still cannot tell why a
24-year-old man with five priors gets a risk of 0.69, or whether the model is fair.

**Idea.** Use a model that a person can read completely: a sparse linear model, a score card,
a small decision tree, or a short rule list. Then the model is its own explanation, the
explanation is exact, and a counterfactual ("what must be different") can be read from it.
The price is paid in feature engineering, and possibly in accuracy, so measure the accuracy.

A stronger L1 penalty removes features from a logistic regression (`interpretable.py`):

```python
lasso = LogisticRegression(penalty="l1", C=c, solver="liblinear")
sparse = make_pipeline(StandardScaler(), lasso).fit(A_train, y_train)
used = int((lasso.coef_[0] != 0).sum())
```

A score card is a sparse linear model on yes/no conditions with coefficients rounded to points:

```python
lasso = LogisticRegression(penalty="l1", C=0.02, solver="liblinear").fit(C_train, y_train)
points = pd.Series(np.round(lasso.coef_[0] / 0.35).astype(int), index=C_train.columns)
```

## What the code shows

- `opaque.py`: the parameters of gradient boosting and of a neural network, and the question
  about one person that they do not answer.
- `interpretable.py`:
  1. Sparse logistic regression: 10 of 15 features give the same accuracy as all 15 (0.674
     against 0.675); with 5 features it is 0.656.
  2. A score card with 9 conditions (e.g. `priors >3` +2, `employed` −2) and a table from
     points to rearrest rate (23% at −2 points, 83% at 4 points).
  3. A decision tree with 6 leaves. Its first split is `age_at_first_arrest`, a feature that is
     not a cause in the data generator: an interpretable model is not a causal model.
  4. Two rule lists with 3 rules: one from the literature (learned with CORELS on real data)
     and one learned by `rule_list.py`.
  5. The prediction for an 18-year-old male with one prior, read from the rule list and the
     score card, with a counterfactual.
  6. **Accuracy.** No model can be better than the true process (0.682, the labels are noisy).
     Logistic regression and the explainable boosting machine are **as accurate as** gradient
     boosting (all 0.675), and the sparse model with 10 features is at 0.674. The score card
     (0.662), the rule lists (0.646, 0.644), and the tree (0.636) lose 1 to 4 points.
- `stability.py`: the same tree learner on 6 samples of the training data gives 6 different
  trees, which agree on only 85% of the predictions on average. A logistic regression with all
  pairs of features has 120 terms, and a tree can have 2 822 leaves: "linear" and "tree" do not
  mean "interpretable".
- `compas_real.py`: the same comparison on the 7 214 real records from Broward County that
  ProPublica published. Gradient boosting 0.677, logistic regression with 6 features 0.676, a
  tree with 6 leaves 0.674, the three-rule list 0.664, the proprietary COMPAS score 0.654.
  The script also fits a logistic regression with all pairs of features: 21 terms such as
  `-0.0031 * age * priors`, accuracy 0.668. It is linear, less accurate than the plain model,
  and nobody can read the effect of one more prior from it.

## Tools

- [scikit-learn](https://scikit-learn.org): the standard machine-learning library for Python.
  Here: logistic regression with L1 penalty, decision trees, gradient boosting, a neural network.
- [InterpretML](https://interpret.ml): a library for interpretable models and explanations.
  Here: the explainable boosting machine, an additive model with one learned curve per feature.
- `rule_list.py`: 30 lines that learn a rule list greedily. Optimal learners such as
  [CORELS](https://github.com/corels/corels) search much longer for a better list.

## Run

With [uv](https://docs.astral.sh/uv/):

```sh
uv run opaque.py          # parameters of opaque models
uv run interpretable.py   # sparse model, score card, tree, rule lists, accuracy table
uv run stability.py       # unstable trees, big linear models, big trees
uv run compas_real.py     # the comparison and the big linear model on real COMPAS data (needs the network)
```
