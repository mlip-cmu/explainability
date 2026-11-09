import pandas as pd
from recidivism_data import FEATURES, TARGET, load, slide_examples
from recidivism_data.generate import make_defendants, make_dirty


def test_deterministic():
    pd.testing.assert_frame_equal(make_defendants(500), make_defendants(500))


def test_clean_data_meets_rules():
    df = load()
    assert df["id"].is_unique
    assert df["age"].between(18, 75).all()
    assert (df["age_at_first_arrest"] <= df["age"]).all()
    assert ((df["priors_count"] == 0) == (df["age_at_first_arrest"] == df["age"])).all()
    assert df["priors_count"].between(0, 38).all()
    assert df[TARGET].isin([0, 1]).all()
    assert 0.35 < df[TARGET].mean() < 0.55
    assert not df[FEATURES].isna().any().any()


def test_every_error_is_logged():
    clean = make_defendants()
    dirty, log = make_dirty(clean)
    diff = (clean != dirty).to_numpy().sum()
    assert diff == len(log)
    assert set(log["id"]) <= set(clean.loc[clean["split"] == "train", "id"])
    assert log["id"].nunique() == 60


def test_slide_examples():
    ex = slide_examples()
    assert list(ex.loc[0, ["age", "sex", "priors_count"]]) == [18, "male", 1]
