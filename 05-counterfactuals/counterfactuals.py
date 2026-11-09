"""Counterfactual explanations: what must be different for a different prediction?"""

import warnings
from itertools import combinations, product

import dice_ml
import pandas as pd
from recidivism_data import FEATURES, NUMERIC, TARGET, load, load_xy
from recidivism_data.shared import opaque_model, person

warnings.filterwarnings("ignore")

X_train, y_train = load_xy("train")
model = opaque_model()
one = person()


def risk(X: pd.DataFrame) -> float:
    return model.predict_proba(X)[:, 1]


def changes(cf: pd.Series) -> str:
    original = one.iloc[0]
    return ", ".join(f"{f}: {original[f]} -> {cf[f]}" for f in FEATURES if cf[f] != original[f])


def problems(cf: pd.Series) -> str:
    original, found = one.iloc[0], []
    if cf["age_at_first_arrest"] > cf["age"]:
        found.append("impossible: first arrest after the current age")
    elif cf["age_at_first_arrest"] != original["age_at_first_arrest"]:
        found.append("the past cannot change")
    if (
        cf["priors_count"] < original["priors_count"]
        or cf["juvenile_priors"] != original["juvenile_priors"]
    ):
        found.append("priors cannot go away")
    if cf["age"] < original["age"]:
        found.append("nobody gets younger")
    if cf["age"] > original["age"]:
        found.append(f"wait {cf['age'] - original['age']} years")
    if cf["sex"] != original["sex"] or cf["county"] != original["county"]:
        found.append("not a reasonable demand")
    if (
        cf["charge_type"] != original["charge_type"]
        or cf["charge_degree"] != original["charge_degree"]
    ):
        found.append("the charge is given")
    return "; ".join(found) or "actionable"


print("1. The prediction")
print("  ", one.iloc[0].to_dict())
print(f"   P(rearrest) = {risk(one)[0]:.2f}: the model predicts a rearrest.")
print("   Question of the defendant: what must be different for the opposite prediction?")

print("\n2. Search by hand: change one feature, then two (growing distance)")
values = {f: sorted(X_train[f].unique()) for f in FEATURES}
found = []
for size in [1, 2]:
    for group in combinations(FEATURES, size):
        if any(set(f) <= set(group) for f, _ in found):
            continue  # a subset of these features is enough already
        candidates = pd.DataFrame(list(product(*(values[f] for f in group))), columns=list(group))
        varied = pd.concat([one] * len(candidates), ignore_index=True).assign(**candidates)
        flipped = varied[risk(varied) < 0.5]
        if len(flipped):
            numeric = [f for f in group if f in NUMERIC]
            step = (
                (flipped[numeric] - one[numeric].iloc[0]).abs().sum(axis=1)
                if numeric
                else flipped.index
            )
            found.append((group, flipped.loc[pd.Series(step, index=flipped.index).idxmin()]))
    print(
        f"   with {size} changed feature(s): {sum(len(g) == size for g, _ in found)} counterfactual(s)"
    )
for group, cf in found:
    print(
        f"   - {changes(cf):<58} P = {risk(cf.to_frame().T.astype(one.dtypes))[0]:.2f}  [{problems(cf)}]"
    )
actionable = sum(problems(cf) == "actionable" for _, cf in found)
print(f"   {len(found)} different smallest changes, {actionable} of them actionable:")
print("   which one is 'the' explanation? (Rashomon effect)")

print("\n3. DiCE: a library that searches for a diverse set of counterfactuals")
data = dice_ml.Data(
    dataframe=X_train.assign(**{TARGET: y_train}), continuous_features=NUMERIC, outcome_name=TARGET
)
dice = dice_ml.Dice(data, dice_ml.Model(model=model, backend="sklearn"), method="random")
result = dice.generate_counterfactuals(
    one, total_CFs=5, desired_class="opposite", random_seed=1, verbose=False
)
for _, cf in result.cf_examples_list[0].final_cfs_df.iterrows():
    print(f"   - {changes(cf):<58} [{problems(cf)}]")
print("   each of them changes the prediction, but most ask for something impossible")

print("\n4. Actionable counterfactuals: only what the defendant can change")
print("   allowed: get a job, start treatment, wait up to 6 years")
genetic = dice_ml.Dice(data, dice_ml.Model(model=model, backend="sklearn"), method="genetic")
result = genetic.generate_counterfactuals(
    one,
    total_CFs=3,
    desired_class="opposite",
    features_to_vary=["employed", "in_treatment", "age"],
    permitted_range={"age": [int(one["age"].iloc[0]), 30]},
    verbose=False,
)
for _, cf in result.cf_examples_list[0].final_cfs_df.iterrows():
    row = cf[FEATURES].to_frame().T.astype(one.dtypes)
    print(f"   - {changes(cf):<58} P = {risk(row)[0]:.2f}")
for label, change in [
    ("job only", {"employed": 1}),
    ("treatment only", {"in_treatment": 1}),
    ("job and treatment", {"employed": 1, "in_treatment": 1}),
]:
    print(f"   check: {label:<18} P = {risk(one.assign(**change))[0]:.2f}")

print("\n5. Sometimes there is no actionable counterfactual")
df = load().query("split == 'test'")
high = df[FEATURES][risk(df[FEATURES]) > 0.5]
best = high.assign(employed=1, in_treatment=1)
share = (risk(best) < 0.5).mean()
print(f"   {len(high)} people in the test data have a predicted rearrest")
print(f"   with a job and treatment, the prediction changes for {share:.0%} of them")
print(f"   for the other {1 - share:.0%}, the only honest answer is: nothing that you can do now")
