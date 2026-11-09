"""Simulation: a true rearrest process, the observed records, and injected data errors."""

import numpy as np
import pandas as pd

SEED = 19
N = 12_000

FEATURES = [
    "age",
    "sex",
    "priors_count",
    "juvenile_priors",
    "age_at_first_arrest",
    "charge_degree",
    "charge_type",
    "employed",
    "in_treatment",
    "county",
]
NUMERIC = ["age", "priors_count", "juvenile_priors", "age_at_first_arrest"]
CATEGORICAL = ["sex", "charge_degree", "charge_type", "county"]
BINARY = ["employed", "in_treatment"]
TARGET = "rearrested"

CHARGE_TYPES = ["drug", "property", "violent", "traffic", "other"]
CHARGE_EFFECT = {
    "drug": 0.3,
    "property": 0.2,
    "violent": -0.1,
    "traffic": -0.5,
    "other": 0.0,
}
COUNTIES = ["north", "south", "east", "west"]


def true_logit(df: pd.DataFrame) -> np.ndarray:
    """Log-odds of a rearrest within two years. The data generator and nothing else knows this.

    Not used: age_at_first_arrest and county. They correlate with the causes, but are no causes.
    """
    age = df["age"].to_numpy(float)
    priors = df["priors_count"].to_numpy(float)
    drug = (df["charge_type"] == "drug").to_numpy()
    return (
        -1.55
        + 1.6 * np.exp(-(age - 18) / 7)
        - 0.015 * (age - 18)
        + 0.75 * np.log1p(priors)
        + 0.25 * np.minimum(df["juvenile_priors"].to_numpy(float), 3)
        + 0.35 * (df["sex"] == "male").to_numpy()
        + 0.15 * (df["charge_degree"] == "felony").to_numpy()
        + df["charge_type"].map(CHARGE_EFFECT).to_numpy(float)
        - 0.5 * df["employed"].to_numpy(float)
        - 0.9 * df["in_treatment"].to_numpy(float) * drug
    )


def true_probability(df: pd.DataFrame) -> np.ndarray:
    return 1 / (1 + np.exp(-true_logit(df)))


def make_defendants(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(18 + rng.gamma(2.0, 7.5, n), 18, 75).round().astype(int)
    sex = rng.choice(["male", "female"], n, p=[0.8, 0.2])
    activity = rng.gamma(1.2, 1.0, n)
    juvenile = np.minimum(rng.poisson(0.35 * activity), 6)
    first_arrest = np.clip(age - rng.gamma(1.5, 4.0, n) * activity, 12, age).round().astype(int)
    first_arrest = np.where(juvenile > 0, np.minimum(first_arrest, 17), first_arrest)
    years_active = age - first_arrest
    priors = np.minimum(rng.poisson(0.45 * activity * np.sqrt(years_active + 0.5)), 38)
    priors = np.where((first_arrest < age) & (priors == 0), 1, priors)
    first_arrest = np.where(priors == 0, age, np.minimum(first_arrest, age - 1))
    charge_type = rng.choice(CHARGE_TYPES, n, p=[0.28, 0.26, 0.18, 0.12, 0.16])
    p_felony = np.select([charge_type == "violent", charge_type == "traffic"], [0.7, 0.1], 0.45)
    charge_degree = np.where(rng.random(n) < p_felony, "felony", "misdemeanor")
    employed = (rng.random(n) < 0.3 + 0.3 * (age > 25) - 0.012 * np.minimum(priors, 15)).astype(int)
    in_treatment = (rng.random(n) < np.where(charge_type == "drug", 0.35, 0.08)).astype(int)
    p_county = np.where((priors > 3)[:, None], [0.15, 0.45, 0.2, 0.2], [0.3, 0.2, 0.25, 0.25])
    county = np.array(COUNTIES)[(rng.random(n)[:, None] > p_county.cumsum(axis=1)).sum(axis=1)]
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "age": age,
            "sex": sex,
            "priors_count": priors,
            "juvenile_priors": juvenile,
            "age_at_first_arrest": first_arrest,
            "charge_degree": charge_degree,
            "charge_type": charge_type,
            "employed": employed,
            "in_treatment": in_treatment,
            "county": county,
        }
    )
    df["true_probability"] = true_probability(df).round(4)
    df[TARGET] = (rng.random(n) < df["true_probability"]).astype(int)
    df["split"] = rng.choice(["train", "valid", "test"], n, p=[0.6, 0.2, 0.2])
    return df


def make_dirty(clean: pd.DataFrame, seed: int = SEED + 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A copy of the training records with data-entry errors, and the log of each error."""
    rng = np.random.default_rng(seed)
    dirty = clean.copy()
    train = clean.index[clean["split"] == "train"].to_numpy()
    log = []

    def change(idx, column, value, kind):
        log.append((int(clean.at[idx, "id"]), column, kind, clean.at[idx, column], value))
        dirty.at[idx, column] = value

    picked = rng.choice(train, 60, replace=False)
    for idx in picked[:20]:
        change(
            idx,
            "priors_count",
            int(clean.at[idx, "priors_count"]) + 90,
            "priors_plus_90",
        )
        if clean.at[idx, TARGET] == 1:
            change(idx, TARGET, 0, "label_flip")
    for idx in picked[20:40]:
        change(idx, "age", int(clean.at[idx, "age"]) + 100, "age_plus_100")
        if clean.at[idx, TARGET] == 0:
            change(idx, TARGET, 1, "label_flip")
    for idx in picked[40:]:
        change(idx, TARGET, 1 - int(clean.at[idx, TARGET]), "label_flip")
    cols = ["id", "column", "kind", "clean_value", "dirty_value"]
    return dirty, pd.DataFrame(log, columns=cols).sort_values(["id", "column"], ignore_index=True)


def slide_examples() -> pd.DataFrame:
    """People from the lecture examples, in the format of the defendants table."""
    rows = [
        (900001, 18, "male", 1, 0, 17, "misdemeanor", "property", 0, 0, "north"),
        (900002, 22, "male", 3, 0, 18, "felony", "drug", 0, 0, "south"),
        (900003, 35, "female", 5, 0, 22, "felony", "property", 1, 0, "east"),
        (900004, 40, "male", 0, 0, 40, "misdemeanor", "traffic", 1, 0, "west"),
    ]
    df = pd.DataFrame(rows, columns=["id", *FEATURES])
    df["true_probability"] = true_probability(df).round(4)
    return df
