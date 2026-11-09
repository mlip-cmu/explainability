# recidivism-data · the shared synthetic dataset

All projects in this repository use one case study: a court uses a model that predicts whether a
defendant is **rearrested within two years**, as an input to bail and sentencing decisions. This
package generates the data for it. The data is **synthetic and seeded**: no real person is in
it, and the process that produces the labels is known. Thus each explanation in the other
projects can be compared with the truth.

## Tables

| Table | Rows | Content |
|---|---|---|
| [`defendants`](data/preview/defendants.csv) | 12 000 | One row per defendant: features, the label `rearrested`, the `true_probability`, and the `split` (train / valid / test). |
| [`defendants_dirty`](data/preview/defendants_dirty.csv) | 12 000 | The same table with data-entry errors in 60 training rows. |
| [`error_log`](data/error_log.csv) | 77 | Each injected error: id, column, kind, clean value, dirty value. |
| [`slide_examples`](data/preview/slide_examples.csv) | 4 | Example people from the lecture, e.g. an 18-year-old male with one prior. |

The complete tables are in `data/*.csv`; `data/preview/` has the first 200 rows of each, which
GitHub shows as a table.

| Feature | Values | In the true process? |
|---|---|---|
| `age` | 18–75 | yes, nonlinear: the risk decreases quickly after age 18 |
| `sex` | male, female | yes |
| `priors_count` | 0–38 | yes, logarithmic |
| `juvenile_priors` | 0–6 | yes, up to 3 |
| `age_at_first_arrest` | 12–75 | **no**, but correlated with age and priors |
| `charge_degree` | felony, misdemeanor | yes, small |
| `charge_type` | drug, property, violent, traffic, other | yes |
| `employed` | 0, 1 | yes |
| `in_treatment` | 0, 1 | yes, **only for drug charges** (an interaction) |
| `county` | north, south, east, west | **no**, but `south` has more people with many priors |

The true process is `true_logit` in [`generate.py`](recidivism_data/generate.py). The label is a
random draw from the true probability, so even the true process predicts only 68% of the test
labels correctly. No model can do better; this is realistic for this task.

The injected errors are `priors_count + 90` (20 rows), `age + 100` (20 rows), and flipped labels
(20 rows, and also on the rows with a wrong value when the label did not fit the wrong value).

## Use

```python
from recidivism_data import load, load_xy, onehot, true_probability
from recidivism_data.shared import opaque_model, person

X_train, y_train = load_xy("train")  # features and label of one split
model = opaque_model()  # gradient boosting, the model that the projects explain
model.predict_proba(person())  # the defendant that the projects explain (id 1657)
```

`onehot` encodes the categories as 0/1 columns. `recidivism_data.theme.use_theme()` sets one
chart style for all notebooks.

## Run

With [uv](https://docs.astral.sh/uv/):

```sh
uv run python -m recidivism_data   # generate the data again (the CSV files do not change)
uv run pytest                      # tests: deterministic, rules hold, each error is logged
```
