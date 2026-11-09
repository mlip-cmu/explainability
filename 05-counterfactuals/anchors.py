"""Anchors: a sufficient condition. 'While these facts hold, the prediction stays the same.'"""

import numpy as np
import pandas as pd
from recidivism_data import FEATURES, load_xy
from recidivism_data.shared import opaque_model, person

X_train, _ = load_xy("train")
model = opaque_model()
one = person().iloc[0]
rng = np.random.default_rng(0)

# Candidate facts about this person: his category, or the range that his number is in.
PREDICATES = {
    "age <= 25": X_train["age"] <= 25,
    "priors_count >= 4": X_train["priors_count"] >= 4,
    "juvenile_priors = 0": X_train["juvenile_priors"] == 0,
    "age_at_first_arrest <= 17": X_train["age_at_first_arrest"] <= 17,
    "sex = male": X_train["sex"] == "male",
    "charge_type = drug": X_train["charge_type"] == "drug",
    "charge_degree = felony": X_train["charge_degree"] == "felony",
    "employed = 0": X_train["employed"] == 0,
    "in_treatment = 0": X_train["in_treatment"] == 0,
    "county = south": X_train["county"] == "south",
}


def precision(anchor: list[str], n: int = 4000) -> float:
    """Share of 'people like him' with the same prediction: the facts of the anchor hold,
    each other feature is a random value from the training data."""
    sample = pd.DataFrame({f: rng.choice(X_train[f].to_numpy(), n) for f in FEATURES})
    for name in anchor:
        feature = name.split(" ")[0]
        sample[feature] = rng.choice(X_train.loc[PREDICATES[name], feature].to_numpy(), n)
    return float(np.mean(model.predict(sample) == 1))


def coverage(anchor: list[str]) -> float:
    return float(np.logical_and.reduce([PREDICATES[name] for name in anchor]).mean())


print("1. The prediction")
print("  ", one.to_dict())
print(f"   P(rearrest) = {model.predict_proba(person())[0, 1]:.2f}: the model predicts a rearrest.")
print(f"   with no fact fixed, {precision([]):.0%} of random people get the same prediction")

print("\n2. Greedy search: add the fact that increases the precision most, until 95%")
anchor: list[str] = []
while precision(anchor) < 0.95 and len(anchor) < len(PREDICATES):
    best = max((p for p in PREDICATES if p not in anchor), key=lambda p: precision([*anchor, p]))
    anchor.append(best)
    print(f"   + {best:<28} precision {precision(anchor):.0%}, coverage {coverage(anchor):.1%}")

print("\n3. The anchor")
print(f"   IF {' AND '.join(anchor)}")
print("   THEN the model predicts a rearrest, whatever the other features are")
print(f"   precision {precision(anchor, 20000):.0%}: how often the rule holds for such people")
print(f"   coverage {coverage(anchor):.1%}: the share of all people to which the rule applies")

print("\n4. Compare with the other explanations of the same prediction")
others = [p for p in PREDICATES if p not in anchor]
print(f"   not necessary for the prediction: {', '.join(others)}")
print("   feature influence (LIME, SHAP): how much each feature pushed")
print("   counterfactual: what to change for a different prediction")
print("   anchor: which facts are enough to keep the prediction")
