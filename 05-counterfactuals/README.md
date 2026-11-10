# 05 · Anchors and counterfactual explanations

A court uses an opaque gradient boosting model that predicts whether a defendant is rearrested
within two years ([dataset and model](../recidivism-data/)). The model gives a 24-year-old
unemployed man with five priors and a drug charge a risk of 0.69: it predicts a rearrest.

**Problem.** A list of feature influences does not answer the two questions that a person asks
about a decision: "Which facts decided my case?" and "What must be different for a different
result?"

**Idea.** An *anchor* is a sufficient condition: a small set of facts about the person such
that the prediction stays the same (with high precision) whatever the other features are. A
*counterfactual* is the smallest change of the input that changes the prediction ("if you had a
job and were in treatment, the prediction would be no rearrest"). Both need only predictions
of the model. There are usually many counterfactuals, and many of them are not possible for
the person, so the search needs constraints.

Search by hand: try all values of one feature, then of two features
(`anchors_and_counterfactuals.ipynb`):

```python
candidates = pd.DataFrame(list(product(*(values[f] for f in group))), columns=list(group))
varied = pd.concat([one] * len(candidates), ignore_index=True).assign(**candidates)
flipped = varied[risk(varied) < 0.5]
```

DiCE searches for diverse counterfactuals; constraints make them actionable:

```python
result = genetic.generate_counterfactuals(
    one,
    total_CFs=3,
    desired_class="opposite",
    features_to_vary=["employed", "in_treatment", "age"],
    permitted_range={"age": [int(one["age"].iloc[0]), 30]},
    verbose=False,
)
```

## What the code shows

Open [`anchors_and_counterfactuals.ipynb`](anchors_and_counterfactuals.ipynb) on GitHub to see
the outputs.

1. **Both ideas in one picture each**, for a model with only the age as a feature. The anchor
   for a 22-year-old is the region `age <= 23` (precision 100%, coverage 16%); the wider region
   `age <= 27` has a precision of only 45%. The counterfactual for a 20-year-old is the age 24,
   and for a 40-year-old the age 23.
2. **An anchor for the model with all features.** A greedy search adds one fact at a time. The
   anchor is `priors_count >= 4 AND age <= 25 AND age_at_first_arrest <= 17 AND employed = 0`:
   97% of such people get the same prediction (precision), and the rule applies to 2.7% of all
   people (coverage). The charge, the sex, and the county are not necessary for this
   prediction.
3. **Counterfactuals for the same prediction.**
   - **Search by hand.** One changed feature is enough in one way (first arrest at 37 in place
     of 17); two changed features give 18 more. Of these 19 smallest changes only one is
     actionable now (a job and treatment, P = 0.41). The others change the past, the sex, or
     the charge, or need a wait of 9 to 46 years.
   - **DiCE without constraints** returns five counterfactuals; four of them describe an
     impossible person (first arrest at 72 for a 54-year-old).
   - **DiCE with constraints** (job, treatment, wait up to 6 years) finds the actionable
     answer. A job alone (P = 0.59) or treatment alone (P = 0.54) is not enough.
   - **No counterfactual.** Of the 678 people in the test data with a predicted rearrest, a job
     and treatment change the prediction for 61%. For the other 39% there is nothing that they
     can do now.

## Tools

- [DiCE](https://github.com/interpretml/DiCE) (`dice-ml`): a library that generates diverse
  counterfactual explanations for any model. Here: the `random` and `genetic` search, with
  `features_to_vary` and `permitted_range` as constraints.
- [scikit-learn](https://scikit-learn.org): the opaque model.
- The anchor search in the notebook is a small implementation of the idea of
  [Anchors](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf). Libraries with the complete
  algorithm are [alibi](https://github.com/SeldonIO/alibi) and
  [anchor-exp](https://github.com/marcotcr/anchor).
- [Matplotlib](https://matplotlib.org) and [Jupyter](https://jupyter.org): the charts and the
  notebook.

## Run

To change the notebook and run it again, with [uv](https://docs.astral.sh/uv/):

```sh
uv run jupyter lab anchors_and_counterfactuals.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace anchors_and_counterfactuals.ipynb
```
