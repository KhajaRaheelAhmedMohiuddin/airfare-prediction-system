"""Streamlit front end for the Airfare Prediction System.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from advisor import BookingAdvisor  # noqa: E402
from config import REPORT_DIR  # noqa: E402
from data_prep import prepare  # noqa: E402
from predict import FarePredictor  # noqa: E402

st.set_page_config(page_title="Airfare Prediction System", page_icon=":airplane:",
                   layout="wide")


@st.cache_resource(show_spinner="Loading the trained ensemble...")
def load_advisor() -> BookingAdvisor:
    return BookingAdvisor(FarePredictor())


@st.cache_data
def load_reference() -> pd.DataFrame:
    train, test = prepare()
    return pd.concat([train.drop(columns=["Price"]), test], ignore_index=True)


advisor = load_advisor()
reference = load_reference()
metrics = advisor.predictor.bundle["metrics"]
DATE_MIN, DATE_MAX = (pd.Timestamp(d) for d in advisor.predictor.date_range)


def check_date(day) -> bool:
    """Warn in the UI when a chosen date sits outside the model's training window."""
    stamp = pd.Timestamp(day)
    if DATE_MIN <= stamp <= DATE_MAX:
        return True
    st.warning(
        f"{stamp:%d %b %Y} is outside the data the model was trained on "
        f"({DATE_MIN:%d %b %Y} to {DATE_MAX:%d %b %Y}). Anything shown below is "
        "extrapolation, not a supported estimate."
    )
    return False

st.title("Airfare Prediction System")
st.caption(
    "Gradient-boosted ensemble trained on 10,463 Indian domestic itineraries. "
    "Prices a flight, then finds you a cheaper way to make the same trip."
)

best = metrics.get("stacked_ensemble", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cross-validated R2", f"{best.get('R2', 0):.4f}")
c2.metric("Accuracy (100 - MAPE)", f"{best.get('Accuracy_100_minus_MAPE', 0):.2f}%")
c3.metric("MAE", f"Rs.{best.get('MAE', 0):,.0f}")
c4.metric("Within 10% of actual", f"{best.get('Within_10pct', 0):.1f}%")

tab_price, tab_cheap, tab_when, tab_perf = st.tabs(
    ["Price a flight", "Cheapest options", "Best travel dates", "Model performance"]
)

AIRLINES = sorted(reference["Airline"].dropna().unique())
SOURCES = sorted(reference["Source_City"].dropna().unique())
DESTS = sorted(reference["Destination_City"].dropna().unique())
STOPS = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
INFO = sorted(reference["Additional_Info"].dropna().unique())

# --------------------------------------------------------------- price a flight
with tab_price:
    st.subheader("Fare estimate for a specific itinerary")
    a, b, c = st.columns(3)
    airline = a.selectbox("Airline", AIRLINES, index=AIRLINES.index("IndiGo")
                          if "IndiGo" in AIRLINES else 0)
    source = b.selectbox("From", SOURCES)
    destination = c.selectbox("To", [d for d in DESTS if d != source])

    d, e, f = st.columns(3)
    journey = d.date_input("Date of journey", value=pd.Timestamp("2019-06-15"))
    dep = e.time_input("Departure", value=pd.Timestamp("09:25").time())
    arr = f.time_input("Arrival", value=pd.Timestamp("13:15").time())

    g, h, i = st.columns(3)
    hours = g.number_input("Duration (hours)", 0, 40, 4)
    minutes = h.number_input("Duration (minutes)", 0, 59, 0)
    stops = i.selectbox("Stops", STOPS, index=1)
    info = st.selectbox("Additional info", INFO,
                        index=INFO.index("No info") if "No info" in INFO else 0)

    if st.button("Estimate fare", type="primary"):
        check_date(journey)
        row = pd.DataFrame([{
            "Airline": airline,
            "Date_of_Journey": journey.strftime("%d/%m/%Y"),
            "Source": source,
            "Destination": destination,
            "Route": f"{source} -> {destination}",
            "Dep_Time": dep.strftime("%H:%M"),
            "Arrival_Time": arr.strftime("%H:%M"),
            "Duration": f"{hours}h {minutes}m",
            "Total_Stops": stops,
            "Additional_Info": info,
        }])
        result = advisor.predictor.predict_with_spread(row).iloc[0]
        st.success(f"Estimated fare: **Rs. {result['predicted_price']:,.0f}**")
        st.caption(
            f"Ensemble range Rs.{result['low_estimate']:,.0f} - "
            f"Rs.{result['high_estimate']:,.0f}. A wide range means the itinerary is "
            "unusual for this route and the estimate is less certain."
        )

# ------------------------------------------------------------ cheapest options
with tab_cheap:
    st.subheader("Cheapest ways to fly this route")
    a, b, c = st.columns(3)
    src2 = a.selectbox("From ", SOURCES, key="src2")
    dst2 = b.selectbox("To ", [d for d in DESTS if d != src2], key="dst2")
    date2 = c.date_input("Travel date", value=pd.Timestamp("2019-06-15"), key="date2")
    max_stops = st.slider("Maximum stops", 0, 3, 2)

    if st.button("Find cheapest options"):
        check_date(date2)
        try:
            options = advisor.cheapest_options(
                src2, dst2, date2.strftime("%d/%m/%Y"), top_n=12, max_stops=max_stops
            )
            view = options[[
                "Airline", "Dep_Time", "Arrival_Time", "Duration", "Total_Stops",
                "predicted_price", "pct_above_cheapest", "saving_vs_market_median",
            ]].round(0)
            st.dataframe(view, use_container_width=True, hide_index=True)
            top = options.iloc[0]
            runner_up_gap = (
                options["predicted_price"].iloc[1] - top["predicted_price"]
                if len(options) > 1
                else 0.0
            )
            confidence = advisor.confidence_for_gap(runner_up_gap, kind="route")
            if confidence:
                st.caption(
                    f"Cheapest vs second cheapest differs by Rs.{runner_up_gap:,.0f}; "
                    f"orderings that close are {confidence} on held-out data."
                )
            st.info(
                f"Cheapest: **{top['Airline']}** departing {top['Dep_Time']} "
                f"({top['Total_Stops']}) at **Rs.{top['predicted_price']:,.0f}** - "
                f"about Rs.{max(top['saving_vs_market_median'], 0):,.0f} below the "
                "median fare on this route."
            )
        except ValueError as exc:
            st.warning(str(exc))

# --------------------------------------------------------- best travel dates
with tab_when:
    st.subheader("When should I fly?")
    a, b, c = st.columns(3)
    src3 = a.selectbox("From  ", SOURCES, key="src3")
    dst3 = b.selectbox("To  ", [d for d in DESTS if d != src3], key="dst3")
    start = c.date_input("Window starts", value=pd.Timestamp("2019-06-01"), key="d3")
    window = st.slider("Days to scan", 5, 27, 14)

    if st.button("Scan dates"):
        check_date(start)
        check_date(pd.Timestamp(start) + pd.Timedelta(days=window - 1))
        dates = advisor.date_range(start.strftime("%d/%m/%Y"), window)
        with st.spinner("Pricing every itinerary on every date..."):
            table = advisor.best_travel_dates(src3, dst3, dates)
        if table.empty:
            st.warning("No itineraries known for that city pair.")
        else:
            chart = table.set_index("date")["cheapest_fare"].sort_index()
            st.line_chart(chart, height=260)
            st.dataframe(table.round(0), use_container_width=True, hide_index=True)
            spread = table["cheapest_fare"].max() - table["cheapest_fare"].min()
            st.success(
                f"Best date: **{table['date'].iloc[0]}** at "
                f"Rs.{table['cheapest_fare'].iloc[0]:,.0f}. Picking the right day in "
                f"this window is worth up to **Rs.{spread:,.0f}**."
            )
            confidence = advisor.confidence_for_gap(spread, kind="date")
            if confidence:
                st.caption(
                    f"That recommendation is {confidence}, measured on held-out data. "
                    "Reliability tracks the size of the gap: differences under "
                    "Rs.250 are close to a coin flip, differences over Rs.2,000 are "
                    "near certain."
                )

# --------------------------------------------------------------- performance
with tab_perf:
    st.subheader("Cross-validated performance (10-fold, out-of-fold predictions)")
    flat = {k: v for k, v in metrics.items() if "R2" in v}
    st.dataframe(pd.DataFrame(flat).T.round(4), use_container_width=True)

    if "fare_bands" in metrics:
        st.subheader("Accuracy by fare band")
        st.dataframe(
            pd.DataFrame(metrics["fare_bands"]).T.round(2), use_container_width=True
        )
        st.caption(
            "Expensive fares are the weak spot: they are rare in the data and the "
            "model still under-predicts them. Treat estimates above Rs.25,000 as "
            "indicative only."
        )

    imp_path = REPORT_DIR / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).head(20).set_index("feature")
        st.subheader("Top 20 features by LightGBM gain")
        st.bar_chart(imp["gain"], height=420)
