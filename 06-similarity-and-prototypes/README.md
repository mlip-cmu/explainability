# 06 · Similar cases, prototypes, and criticisms

A court uses a model that predicts whether a defendant is rearrested within two years
([dataset and model](../recidivism-data/)). The training data is a table of 7 253 past
defendants; a dirty copy of it has data-entry errors in 60 rows.

**Problem.** Feature weights are abstract. People often understand a decision better from
examples ("these five past cases were like yours"), and a developer who wants to understand a
data set cannot read 7 253 rows.

**Idea.** Explain with data points. *Similar cases:* show the nearest neighbors of the person
and their outcomes. *Prototypes:* select a few real people that represent the data.
*Criticisms:* show the people that the prototypes do not represent; errors and rare groups are
there. All three need a distance between people, and are only as good as that distance.

k-medoids gives cluster centers that are real people; the criticisms are the people far from
each prototype (`similarity_and_prototypes.ipynb`):

```python
distances = pairwise_distances(scaled(sample))
clusters = kmedoids.fasterpam(distances, 6, random_state=0)
prototypes = sample.loc[clusters.medoids, FEATURES].reset_index(drop=True)

to_prototype = dirty_distances[:, dirty_clusters.medoids].min(axis=1)
```

## What the code shows

Open [`similarity_and_prototypes.ipynb`](similarity_and_prototypes.ipynb) on GitHub to see
the outputs.

1. **Similar cases.** Of the 25 people most similar to the defendant, 84% were rearrested (the
   opaque model says 0.69). All of the nearest six are from the same county, because each
   feature has the same weight in the distance. Without the two features that are not causes,
   the rate is 76%, and only 6 of the 25 people are the same.
2. **Prototypes.** Six real people represent 3 000: five are about 30 years old with one prior,
   one stands for the people with more priors (61% of that group were rearrested). A plot over
   age and number of priors shows five prototypes and five criticisms; the largest prototype
   (age 25, one prior) stands for 36% of the people.
3. **Criticisms find data errors.** In the dirty data, all 14 sampled rows with an impossible
   value (`priors_count` of 91, `age` of 134) are among the 20 criticisms. Of the 7 rows with
   only a flipped label, none is: criticisms look at features, not at labels.
4. **MMD-critic** selects prototypes whose distribution is near the distribution of the data,
   and criticisms where the two differ most (40 lines of NumPy).

## Tools

- [kmedoids](https://github.com/kno10/python-kmedoids): fast k-medoids clustering (FasterPAM).
  Here: the prototypes.
- [scikit-learn](https://scikit-learn.org): `NearestNeighbors` for the similar cases,
  `pairwise_distances` and `rbf_kernel` for the prototypes, `StandardScaler` for the distance.
- [Matplotlib](https://matplotlib.org) and [Jupyter](https://jupyter.org): chart and notebook.

## Run

With [uv](https://docs.astral.sh/uv/):

```sh
uv run jupyter nbconvert --to notebook --execute --inplace similarity_and_prototypes.ipynb
```
