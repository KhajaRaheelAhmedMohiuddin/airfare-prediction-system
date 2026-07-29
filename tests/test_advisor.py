"""Booking advisor behaviour, including the shapes callers rely on.

Both tests here are regressions for bugs that reached the running app.
"""

import pytest

from config import MODEL_DIR

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "airfare_model.joblib").exists(),
    reason="no trained model; run python src/train.py first",
)


@pytest.fixture(scope="module")
def advisor():
    from advisor import BookingAdvisor

    return BookingAdvisor()


def test_unflown_city_pair_returns_an_empty_frame_not_a_keyerror(advisor):
    """Regression: every date being skipped left an empty frame with no columns,
    so sort_values('cheapest_fare') raised KeyError before the caller could check
    .empty. Cochin is only served from Delhi, so this pair has no itineraries."""
    from advisor import DATE_SCAN_COLUMNS

    result = advisor.best_travel_dates(
        "Banglore", "Cochin", advisor.date_range("15/06/2019", 3)
    )
    assert result.empty
    assert list(result.columns) == DATE_SCAN_COLUMNS


def test_unflown_city_pair_raises_a_clear_error_for_single_date_search(advisor):
    with pytest.raises(ValueError, match="No itineraries known"):
        advisor.cheapest_options("Banglore", "Cochin", "15/06/2019")


def test_real_city_pair_returns_ranked_dates(advisor):
    result = advisor.best_travel_dates(
        "Delhi", "Cochin", advisor.date_range("15/06/2019", 4)
    )
    assert len(result) == 4
    assert result["cheapest_fare"].is_monotonic_increasing
    assert (result["saving_vs_worst_date"] >= 0).all()


def test_options_are_distinct_itineraries(advisor):
    """Regression: rows differing only by the arrival date suffix are the same
    flight at the same predicted fare, and padded the results list."""
    options = advisor.cheapest_options("Banglore", "Delhi", "15/06/2019", top_n=15)
    # Additional_Info is part of the identity: the same departure sold with and
    # without check-in baggage is two real products at two real prices, so the
    # displayed table has to show that column for the rows to make sense.
    key = options[
        ["Airline", "Dep_Time", "Arrival_Time", "Duration", "Total_Stops",
         "Route", "Additional_Info"]
    ]
    assert not key.duplicated().any()
    assert not options["Arrival_Time"].str.contains(" ").any()


def test_cheapest_options_are_sorted_and_annotated(advisor):
    options = advisor.cheapest_options("Delhi", "Cochin", "15/06/2019", top_n=5)
    assert len(options) <= 5
    assert options["predicted_price"].is_monotonic_increasing
    assert options["pct_above_cheapest"].iloc[0] == 0.0


def test_every_offered_city_pair_can_actually_be_priced(advisor):
    """The app builds its dropdowns from city_pairs(); each must be searchable."""
    pairs = advisor.city_pairs()
    assert len(pairs) > 0
    for _, row in pairs.iterrows():
        options = advisor.cheapest_options(
            row["Source"], row["Destination"], "15/06/2019", top_n=1
        )
        assert not options.empty
