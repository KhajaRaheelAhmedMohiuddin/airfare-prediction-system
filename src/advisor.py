"""Booking advisor: turns fare predictions into cost-saving recommendations.

The model prices an itinerary. What a traveller actually wants is the *cheapest
reasonable* itinerary. This module builds a catalogue of real itinerary templates
observed in the data (airline + route + schedule + stops for each city pair), then
re-prices those templates across candidate travel dates and ranks them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import REPORT_DIR
from data_prep import prepare
from predict import FarePredictor

DATE_SCAN_COLUMNS = [
    "date",
    "cheapest_fare",
    "avg_top5_fare",
    "best_airline",
    "best_departure",
    "stops",
    "saving_vs_worst_date",
]

TEMPLATE_COLS = [
    "Airline",
    "Source",
    "Destination",
    "Route",
    "Dep_Time",
    "Arrival_Time",
    "Duration",
    "Total_Stops",
    "Additional_Info",
]


class BookingAdvisor:
    def __init__(self, predictor: FarePredictor | None = None) -> None:
        self.predictor = predictor or FarePredictor()
        train, test = prepare(drop_duplicates=True)
        catalogue = pd.concat(
            [train[TEMPLATE_COLS], test[TEMPLATE_COLS]], ignore_index=True
        )
        catalogue["Source"] = catalogue["Source"].astype(str)
        catalogue["Destination"] = catalogue["Destination"].astype(str)

        # Arrival_Time carries the source row's own journey date ("01:00 04 Jun").
        # Re-priced on some other date that suffix is meaningless, and the features
        # read only the clock portion, so rows differing solely by it are the same
        # bookable option at the same predicted fare. Left in, one Banglore -> Delhi
        # search returned 236 "options" that were really 65, the list padded with
        # visually identical rows.
        catalogue["Arrival_Time"] = (
            catalogue["Arrival_Time"].astype(str).str.split(" ").str[0]
        )
        self.catalogue = catalogue.drop_duplicates().reset_index(drop=True)

    # ------------------------------------------------------------- helpers
    def city_pairs(self) -> pd.DataFrame:
        return (
            self.catalogue.groupby(["Source", "Destination"])
            .size()
            .reset_index(name="itineraries")
            .sort_values("itineraries", ascending=False)
        )

    def _templates(self, source: str, destination: str) -> pd.DataFrame:
        mask = (self.catalogue["Source"] == source) & (
            self.catalogue["Destination"] == destination
        )
        return self.catalogue[mask].reset_index(drop=True)

    # ---------------------------------------------------------- core search
    def cheapest_options(
        self,
        source: str,
        destination: str,
        date_of_journey: str,
        top_n: int = 10,
        max_stops: int | None = None,
        airlines: list[str] | None = None,
    ) -> pd.DataFrame:
        """Rank every known itinerary on a city pair for one travel date."""
        templates = self._templates(source, destination)
        if templates.empty:
            raise ValueError(f"No itineraries known for {source} -> {destination}")

        if airlines:
            templates = templates[templates["Airline"].isin(airlines)]
        if max_stops is not None:
            allowed = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"][
                : max_stops + 1
            ]
            templates = templates[templates["Total_Stops"].isin(allowed)]
        if templates.empty:
            raise ValueError("No itineraries match those filters")

        candidates = templates.copy()
        candidates["Date_of_Journey"] = date_of_journey
        priced = self.predictor.predict_with_spread(candidates)

        out = candidates.reset_index(drop=True).join(priced)
        out = out.sort_values("predicted_price").head(top_n).reset_index(drop=True)
        cheapest = out["predicted_price"].min()
        median_market = priced["predicted_price"].median()
        out["saving_vs_market_median"] = median_market - out["predicted_price"]
        out["pct_above_cheapest"] = (
            (out["predicted_price"] - cheapest) / cheapest * 100
        ).round(1)
        return out

    def best_travel_dates(
        self,
        source: str,
        destination: str,
        dates: list[str],
        airlines: list[str] | None = None,
        max_stops: int | None = None,
    ) -> pd.DataFrame:
        """Cheapest achievable fare on each candidate date - the 'when to fly' view."""
        rows = []
        for date in dates:
            try:
                options = self.cheapest_options(
                    source,
                    destination,
                    date,
                    top_n=5,
                    airlines=airlines,
                    max_stops=max_stops,
                )
            except ValueError:
                continue
            rows.append(
                {
                    "date": date,
                    "cheapest_fare": options["predicted_price"].iloc[0],
                    "avg_top5_fare": options["predicted_price"].head(5).mean(),
                    "best_airline": options["Airline"].iloc[0],
                    "best_departure": options["Dep_Time"].iloc[0],
                    "stops": options["Total_Stops"].iloc[0],
                }
            )
        if not rows:
            # No itinerary exists on this city pair (or none survives the filters).
            # Return the right shape rather than an empty frame, so callers can test
            # .empty instead of catching a KeyError from sort_values.
            return pd.DataFrame(columns=DATE_SCAN_COLUMNS)

        result = pd.DataFrame(rows).sort_values("cheapest_fare").reset_index(drop=True)
        worst = result["cheapest_fare"].max()
        result["saving_vs_worst_date"] = (worst - result["cheapest_fare"]).round(0)
        return result

    def confidence_for_gap(self, gap: float, kind: str = "route") -> str | None:
        """How much to trust "A is cheaper than B" given the predicted gap.

        Backed by measured out-of-fold ranking accuracy, not intuition: a gap under
        Rs.250 is barely better than a coin flip, a gap over Rs.2,000 is near certain.
        """
        path = REPORT_DIR / "savings_confidence_bands.csv"
        if not path.exists():
            return None
        table = pd.read_csv(path)
        wanted = (
            "same route + date, different itinerary"
            if kind == "route"
            else "same flight, different travel date"
        )
        table = table[table["comparison"] == wanted]
        edges = [0, 250, 500, 1000, 2000, np.inf]
        for i, label in enumerate(table["predicted_gap"]):
            if edges[i] <= abs(gap) < edges[i + 1]:
                pct = table.iloc[i]["ranking_accuracy_pct"]
                return f"{pct:.0f}% reliable (gaps of {label})"
        return None

    def reliability(self) -> dict[str, float] | None:
        """Measured confidence in the *comparisons* this advisor makes.

        Loaded from outputs/reports/savings_validation.csv, produced by
        `python src/validate_savings.py`. Returns None if that has not been run.
        """
        path = REPORT_DIR / "savings_validation.csv"
        if not path.exists():
            return None
        table = pd.read_csv(path).set_index("comparison")
        return {
            "route_ranking_accuracy": float(
                table.loc["same route + date, different itinerary", "ranking_accuracy"]
            ),
            "date_ranking_accuracy": float(
                table.loc["same flight, different travel date", "ranking_accuracy"]
            ),
        }

    def date_range(self, start: str, days: int) -> list[str]:
        """dd/mm/yyyy strings for `days` consecutive dates starting at `start`."""
        first = pd.to_datetime(start, format="%d/%m/%Y")
        return [
            (first + pd.Timedelta(days=i)).strftime("%d/%m/%Y") for i in range(days)
        ]


def _demo() -> None:
    advisor = BookingAdvisor()
    src, dst, date = "Delhi", "Cochin", "15/06/2019"

    print(f"\nCheapest options  {src} -> {dst}  on {date}")
    print("-" * 92)
    opts = advisor.cheapest_options(src, dst, date, top_n=8)
    view = opts[
        ["Airline", "Dep_Time", "Duration", "Total_Stops", "predicted_price",
         "pct_above_cheapest"]
    ].copy()
    view["predicted_price"] = view["predicted_price"].round(0)
    print(view.to_string(index=False))

    print(f"\nBest travel dates  {src} -> {dst}  (14-day window from {date})")
    print("-" * 92)
    dates = advisor.date_range(date, 14)
    best = advisor.best_travel_dates(src, dst, dates)
    best["cheapest_fare"] = best["cheapest_fare"].round(0)
    best["avg_top5_fare"] = best["avg_top5_fare"].round(0)
    print(best.to_string(index=False))
    spread = best["cheapest_fare"].max() - best["cheapest_fare"].min()
    print(f"\nShifting the travel date inside this window is worth up to Rs.{spread:,.0f}"
          f" ({spread / best['cheapest_fare'].max() * 100:.1f}% of the priciest day).")

    confidence = advisor.confidence_for_gap(spread, kind="date")
    if confidence:
        print(f"That date-shift recommendation is {confidence} - measured on held-out "
              "data, not assumed.")
    else:
        print("Run `python src/validate_savings.py` to measure how reliable these "
              "comparisons actually are.")


if __name__ == "__main__":
    _demo()
