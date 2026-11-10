"""Is the problem in the model or in the data? A processing error in the real COMPAS data."""

import sys
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
CUTOFF = pd.Timestamp("2014-04-01")  # two years before the end of the data collection

try:
    raw = pd.read_csv(StringIO(requests.get(URL, timeout=30).text))
except (requests.RequestException, pd.errors.ParserError) as error:
    print(f"no download possible ({type(error).__name__}); this script needs the network")
    sys.exit(0)

raw["screened"] = pd.to_datetime(raw["compas_screening_date"])
y = raw["two_year_recid"]

print("1. The data set as published")
print(f"   {len(raw)} defendants from Broward County, screened with COMPAS in 2013 and 2014")
print(f"   rearrested within two years: {y.mean():.1%}")

print("\n2. Look at the data: people by the quarter of their screening and by the outcome")
quarter = raw["screened"].dt.to_period("Q").astype(str)
table = pd.crosstab(quarter, y.map({0: "not rearrested", 1: "rearrested"}))
table["share rearrested"] = (table["rearrested"] / table.sum(axis=1)).map("{:.0%}".format)
print("   " + table.to_string().replace("\n", "\n   "))
late = raw["screened"] > CUTOFF
print(
    f"   after {CUTOFF.date()}: {int((late & (y == 1)).sum())} rearrested, {int((late & (y == 0)).sum())} not rearrested"
)

print("\n3. The cause")
print("   'Not rearrested within two years' is only known for a person who was observed for two")
print("   years, so people screened after 1 April 2014 were removed. But this filter was applied")
print(
    "   only to the people without a rearrest. People with a rearrest were kept to the end of 2014."
)

print("\n4. The same filter for both groups")
kept = raw[~late]
print(f"   {len(kept)} defendants remain ({int(late.sum())} removed)")
print(
    f"   rearrested within two years: {kept['two_year_recid'].mean():.1%}  (published data: {y.mean():.1%})"
)

print("\n5. A model can find such an error, if you let it see the column and then ask it")
X = pd.DataFrame(
    {
        "age": raw["age"],
        "priors": raw["priors_count"],
        "male": (raw["sex"] == "Male").astype(int),
        "screening day": (raw["screened"] - raw["screened"].min()).dt.days,
    }
)
for label, columns in [
    ("age, priors, sex", ["age", "priors", "male"]),
    ("... and the screening day", list(X.columns)),
]:
    X_fit, X_val, y_fit, y_val = train_test_split(X[columns], y, test_size=0.2, random_state=42)
    model = HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, learning_rate=0.05, random_state=0
    )
    model.fit(X_fit, y_fit)
    print(f"   features {label}: accuracy {model.score(X_val, y_val):.3f}")
importance = permutation_importance(model, X_val, y_val, n_repeats=20, random_state=0)
for name, value in sorted(zip(columns, importance.importances_mean), key=lambda item: -item[1]):
    print(f"      permutation importance of {name:<14}{value:.3f}")
at_end = model.predict_proba(X_val[X_val["screening day"] > (CUTOFF - raw["screened"].min()).days])[
    :, 1
]
print(
    f"   predicted risk of the validation people screened after the cutoff: {np.mean(at_end):.2f} on average"
)
print("   The day of the screening cannot cause a rearrest. A high importance of such a feature")
print("   is a sign of a problem in the data, not a finding about people.")
