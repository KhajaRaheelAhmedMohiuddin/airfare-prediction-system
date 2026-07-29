"""The leakage guarantees the reported score depends on."""

import numpy as np
import pandas as pd

from encoders import LabelEncoder, OutOfFoldTargetEncoder


def test_a_rows_own_target_never_reaches_its_own_encoding():
    """The core no-leakage property.

    Change one row's target to an absurd value. If that row's encoded feature moves,
    the encoder is feeding the answer back into the model and every CV number in the
    project is inflated.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"key": ["A", "B", "C", "D"] * 50})
    y = rng.normal(10, 1, size=len(X))

    encoder = OutOfFoldTargetEncoder(["key"], seed=1)
    before = encoder.fit_transform(X, y)["TE_key"].to_numpy()

    poisoned = y.copy()
    poisoned[7] = 10_000.0
    after = OutOfFoldTargetEncoder(["key"], seed=1).fit_transform(X, poisoned)
    after = after["TE_key"].to_numpy()

    assert np.isclose(before[7], after[7]), "row 7 saw its own target"


def test_unseen_categories_fall_back_to_the_prior():
    X = pd.DataFrame({"key": ["A"] * 20 + ["B"] * 20})
    y = np.concatenate([np.full(20, 5.0), np.full(20, 9.0)])

    encoder = OutOfFoldTargetEncoder(["key"], seed=1)
    encoder.fit_transform(X, y)

    unseen = encoder.transform(pd.DataFrame({"key": ["ZZZ"]}))
    assert np.isclose(unseen["TE_key"].iloc[0], y.mean())


def test_smoothing_pulls_small_groups_toward_the_prior():
    """A category seen once should barely move off the global mean."""
    X = pd.DataFrame({"key": ["common"] * 200 + ["rare"]})
    y = np.concatenate([np.full(200, 5.0), np.array([100.0])])

    encoder = OutOfFoldTargetEncoder(["key"], smoothing=20.0, seed=1)
    encoder.fit_transform(X, y)

    rare = encoder.maps_["key"]["rare"]
    common = encoder.maps_["key"]["common"]
    prior = y.mean()
    assert abs(rare - prior) < abs(100.0 - prior) * 0.2
    assert abs(common - 5.0) < 0.5


def test_counts_are_off_by_default_and_optional():
    X = pd.DataFrame({"key": ["A", "B"] * 10})
    y = np.arange(20, dtype=float)

    plain = OutOfFoldTargetEncoder(["key"], seed=1).fit_transform(X, y)
    assert list(plain.columns) == ["TE_key"]

    with_counts = OutOfFoldTargetEncoder(
        ["key"], seed=1, with_counts=True
    ).fit_transform(X, y)
    assert list(with_counts.columns) == ["TE_key", "TEN_key"]


def test_label_encoder_is_stable_and_handles_unseen_values():
    a = pd.DataFrame({"c": ["x", "y"]})
    b = pd.DataFrame({"c": ["z"]})

    encoder = LabelEncoder(["c"]).fit([a, b])
    assert encoder.transform(a)["c"].tolist() == [0, 1]
    assert encoder.transform(b)["c"].tolist() == [2]
    assert encoder.transform(pd.DataFrame({"c": ["brand new"]}))["c"].iloc[0] == -1
