"""Synthetic recidivism data: one case study for all projects in this repository."""

from pathlib import Path

import pandas as pd

from .generate import (
    BINARY,
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    TARGET,
    make_defendants,
    make_dirty,
    slide_examples,
    true_logit,
    true_probability,
)

DATA = Path(__file__).resolve().parent.parent / "data"
TABLES = ["defendants", "defendants_dirty", "error_log"]

__all__ = [
    "BINARY",
    "CATEGORICAL",
    "FEATURES",
    "NUMERIC",
    "TARGET",
    "build",
    "load",
    "load_xy",
    "onehot",
    "slide_examples",
    "true_logit",
    "true_probability",
]


def build(force: bool = False) -> None:
    """Generate all tables: Parquet as a cache, CSV and a preview for readers on GitHub."""
    if not force and all((DATA / f"{t}.parquet").exists() for t in TABLES):
        return
    clean = make_defendants()
    dirty, log = make_dirty(clean)
    log = log.astype({"clean_value": int, "dirty_value": int})
    (DATA / "preview").mkdir(parents=True, exist_ok=True)
    for name, table in zip(TABLES, [clean, dirty, log]):
        table.to_parquet(DATA / f"{name}.parquet", index=False)
        table.to_csv(DATA / f"{name}.csv", index=False)
        table.head(200).to_csv(DATA / "preview" / f"{name}.csv", index=False)
    slide_examples().to_csv(DATA / "preview" / "slide_examples.csv", index=False)


def load(table: str = "defendants") -> pd.DataFrame:
    build()
    return pd.read_parquet(DATA / f"{table}.parquet")


def load_xy(split: str, table: str = "defendants") -> tuple[pd.DataFrame, pd.Series]:
    """Features and label of one split (train, valid, or test)."""
    df = load(table)
    df = df[df["split"] == split].reset_index(drop=True)
    return df[FEATURES], df[TARGET]


def onehot(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix with 0/1 columns for the categories (stable column set)."""
    out = X[NUMERIC + BINARY].astype(float).copy()
    out["male"] = (X["sex"] == "male").astype(float)
    out["felony"] = (X["charge_degree"] == "felony").astype(float)
    for value in ["drug", "property", "violent", "traffic"]:
        out[f"charge_{value}"] = (X["charge_type"] == value).astype(float)
    for value in ["south", "east", "west"]:
        out[f"county_{value}"] = (X["county"] == value).astype(float)
    return out
