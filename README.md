# Airfare Prediction System

A machine-learning system that predicts the fare of an Indian domestic flight from its
itinerary, and then uses those predictions to recommend **cost-effective bookings** —
the cheapest airline/departure combination on a route, and the cheapest date to travel
inside a window.

Built on 10,683 real itineraries (March–June 2019) covering 12 airlines, 5 origin
cities and 129 routes.

---

## Results

**10-fold cross-validated, out-of-fold. The supplied `Test_set.xlsx` has no fares in
it, so it cannot be scored — every number below comes from held-out folds of the
training data.**

| Model | R² | RMSE (₹) | MAE (₹) | MAPE | Accuracy (100−MAPE) | Within 10% |
|---|---|---|---|---|---|---|
| LightGBM | 0.9136 | 1359 | 590 | 6.40% | 93.60% | 81.4% |
| XGBoost | 0.9110 | 1379 | 598 | 6.86% | 93.14% | 79.2% |
| CatBoost | 0.9107 | 1382 | 588 | 6.37% | 93.63% | 81.2% |
| Random Forest | 0.9035 | 1436 | 564 | 6.11% | 93.89% | 81.5% |
| Extra Trees | 0.9130 | 1364 | 526 | 5.70% | **94.30%** | 82.8% |
| Simple average | 0.9151 | 1348 | 548 | 6.00% | 94.00% | 82.9% |
| **Stacked ensemble (deployed)** | **0.9165** | **1337** | **542** | **5.86%** | **94.14%** | **83.5%** |

The deployed model explains **91.7% of fare variance**, is off by **₹542 on average**
against a mean fare of ₹9,027, and lands **within 10% of the true fare on 83.5% of
bookings**.

### Accuracy by fare band

The headline is carried by ordinary tickets. Splitting it honestly:

| Fare band | Rows | MAPE | Signed error | Mean actual | Mean predicted |
|---|---|---|---|---|---|
| ≤ ₹25,000 | 10,381 | 5.78% | +0.61% | ₹8,850 | ₹8,832 |
| > ₹25,000 | 82 | 15.91% | **−12.30%** | ₹31,390 | ₹26,381 |

Expensive fares are systematically under-predicted and there is no fixing it with 82
examples. See "what was tried and rejected" below.

### Which metric wins depends on the metric

The ridge stack is best on R²/RMSE. Extra Trees alone is best on percentage accuracy
(94.30% vs 94.14%), because squared error chases expensive outliers while percentage
error chases the typical booking. The pipeline fits both a ridge stack and a
rupee-MAE-optimised blend, scores them out-of-fold, and deploys the R²/RMSE winner.
Both are trained and reported, so switching preference is a one-line change.

### What was tried and rejected

Three things that looked good and were dropped after measurement — recorded here so
they don't get re-attempted:

1. **Flight-identity target encoding.** Encoding at flight+date granularity is very
   tempting: 60% of test rows share a flight *and* date with a training row. It hurt
   every base learner (R² 0.9179 → 0.9163). Out-of-fold those keys have a group size
   of one and collapse to the global prior, so the model learns to ignore a feature
   that carries real signal at inference — a train/serve mismatch, not leakage.
2. **Isotonic calibration for the high-fare bias.** Measured across five CV seeds, it
   reduces signed bias on fares above ₹25,000 by a stable +2.3pp, but leaves the
   *error* on those fares unchanged (high-band MAPE +0.05pp, sd 0.15) while costing
   −0.0015 R² overall. It relabels the bias without predicting the tail any better.
   The code fits it, scores it, and a gate rejects it — the machinery stays so the
   trade-off is visible rather than becoming folklore.
3. **Shrinking the forests to fit a size budget.** Dropping Extra Trees from 500
   trees at `min_samples_leaf=2` to 400 at 3 saves 25 MB and costs ~0.010 of R². The
   accurate settings were kept and the artifact was split across files instead.

Top features by gain: `TE_Airline_Route`, `TE_Airline_Stops`, `Journey_DayOfYear`,
`TE_Route`, `Journey_WeekOfYear`, `No_Meal`, `Additional_Info`, `Journey_Day`.
Which carrier flies which route dominates everything else.

---

## Cost-effective bookings

Predicting a fare is only half the job — the point is to spend less. `advisor.py` takes
the catalogue of itineraries that actually exist on a city pair, re-prices all of them
with the model, and ranks them. Delhi → Cochin on 15 June 2019:

```
 Airline Dep_Time Duration Total_Stops  predicted_price  pct_above_cheapest
SpiceJet    08:45   4h 30m      1 stop           4960.0                 0.0
   GoAir    10:35       9h      1 stop           5058.0                 2.0
  IndiGo    07:35   4h 35m      1 stop           5102.0                 2.9
```

The same machinery answers "when should I fly?" by scanning a date window. Over the
14 days from 15 June, the cheapest achievable Delhi → Cochin fare ranges from ₹4,437
to ₹5,035 — moving the trip saves about ₹598, roughly 12% of the fare.

### How much can you trust that?

A ₹598 saving from a model whose average error is ₹542 deserves scrutiny, so
`validate_savings.py` measures it directly on held-out predictions rather than
assuming. Two findings:

**Errors do not cancel.** The error on a *difference* between two itineraries is about
1.5× the error on a single fare — slightly worse than the 1.41× you would get if
errors were independent. Comparing two predictions is not safer than making one.

**But reliability tracks the size of the gap**, and that is actionable:

| Predicted gap | Same route, different flight | Same flight, different date |
|---|---|---|
| under ₹250 | 66.5% | 62.1% |
| ₹250–500 | 81.5% | 75.0% |
| ₹500–1,000 | 87.6% | 86.5% |
| ₹1,000–2,000 | 94.9% | 92.0% |
| over ₹2,000 | 97.4% | 98.7% |

Overall the model ranks two itineraries on a route correctly 87.7% of the time, and
two dates for the same flight 85.6% of the time. So the ₹598 date-shift above is
**86% reliable** — worth acting on, but not a certainty. A recommendation resting on a
gap under ₹250 is close to a coin flip and should be ignored. The advisor and the app
both print the measured confidence next to the recommendation, and each option also
carries a `model_spread` column showing how much the five base learners disagree.

---

## How it works

```
data/raw/*.xlsx
      |
      v
 data_prep.py     clean text, normalise "New Delhi" -> "Delhi", repair the one
      |           missing Route/Total_Stops, drop 220 exact duplicate rows
      v
 features.py      61 engineered features (see below)
      |
      v
 encoders.py      out-of-fold target encoding for high-cardinality categories
      |
      v
 models.py        5 base learners: LightGBM, XGBoost, CatBoost, Random Forest,
      |           Extra Trees -- all trained on log1p(Price)
      v
 train.py         10-fold CV -> out-of-fold predictions -> RidgeCV stack
      |
      v
 predict.py       FarePredictor: score any itinerary
 advisor.py       BookingAdvisor: rank itineraries and dates by cost
 app.py           Streamlit UI
```

### Feature engineering (61 features)

| Group | Features |
|---|---|
| **Journey date** | day, month, day-of-week, day-of-year, week-of-year, in-month position, weekend flag, month start/end |
| **Seasonality** | days to the nearest Indian festival/public holiday in the window, holiday-week flag |
| **Departure / arrival** | minutes past midnight, hour, minute, named time slot (red-eye … night), sine/cosine cyclical encodings so 23:50 sits next to 00:10 |
| **Duration** | total minutes, hours, log duration, overnight-crossing flag |
| **Stops** | numeric stop count, direct flag, duration per leg, implied layover time |
| **Route** | hop count, each intermediate airport as its own feature (`Route_Leg_1..4`) |
| **Interactions** | `Airline × Route`, `Airline × Stops`, `Source × Destination` |
| **Service flags** | premium/business cabin, low-cost carrier, no meal, no check-in baggage, long layover, airport change |
| **Market aggregates** | how many itineraries exist per airline/route/city-pair, how this flight's duration compares to the route median and to the fastest option on the route, flights departing that day |
| **Target statistics** | smoothed out-of-fold mean fare for airline, route, airline×route, city pair, airline×stops, first stopover |

### Why the score is trustworthy

Four things that are easy to get wrong here, and how they are handled:

1. **Target encoding leaks.** Mean-fare-by-route is the single strongest feature, and
   computing it on all of the training data inflates CV scores badly. `OutOfFoldTargetEncoder`
   fits inside each fold and uses inner folds for the rows it was fitted on, so no row
   ever contributes to its own encoded value.
2. **Early stopping leaks.** Boosters that early-stop on the fold's validation set
   tune themselves to the very rows they are scored on. Each booster gets a 10% inner
   split carved out of the fold's *training* rows instead.
3. **Duplicate rows inflate scores.** 220 rows are byte-identical repeats including the
   fare; left in, they appear on both sides of a fold split and act as free answers.
   They are dropped before training.
4. **Even the fallback prior leaks.** When a category is unseen, the encoder falls back
   to the mean target — and the global mean includes the row being encoded. It is a
   1/n share, but it is not zero, so the prior is recomputed from the inner-training
   rows of each fold. `tests/test_encoders.py` pins this down by poisoning one row's
   target and asserting that row's own encoded value does not move; that test caught
   the bug.

The test workbook has no labels, so every number reported here is 10-fold
cross-validated on the training data — not a single lucky holdout.

---

## Usage

Install:

```bash
pip install -r requirements.txt
```

Train. **This step is required after cloning** — the trained model files are ~100 MB
of binaries and are deliberately not committed, so `models/` starts empty. This writes
them, along with the metrics reports and `outputs/submission.xlsx` (about 10 minutes
on a laptop):

```bash
python src/train.py
```

Exploratory figures and the data profile:

```bash
python src/eda.py
```

Score the supplied test workbook, or any file with the same columns:

```bash
python src/predict.py
```

Booking recommendations from the command line:

```bash
python src/advisor.py
```

Interactive app:

```bash
python -m streamlit run app.py
```

### As a library

```python
import sys; sys.path.insert(0, "src")
from predict import FarePredictor

p = FarePredictor()
p.predict_one(
    airline="IndiGo", source="Delhi", destination="Cochin",
    date_of_journey="15/06/2019", dep_time="09:25", arrival_time="13:15",
    duration="3h 50m", total_stops="1 stop",
)
```

```python
from advisor import BookingAdvisor

advisor = BookingAdvisor()
advisor.cheapest_options("Delhi", "Cochin", "15/06/2019", top_n=10)
advisor.best_travel_dates("Delhi", "Cochin", advisor.date_range("01/06/2019", 14))
```

---

## Project layout

```
Airfare Prediction System/
├── app.py                     Streamlit UI (price, cheapest options, best dates)
├── requirements.txt
├── data/raw/                  the three supplied workbooks
├── src/
│   ├── config.py              paths, seed, fold count, feature lists
│   ├── data_prep.py           loading and cleaning
│   ├── features.py            feature engineering
│   ├── encoders.py            out-of-fold target encoding, label encoding
│   ├── models.py              the five base learners
│   ├── metrics.py             R2 / RMSE / MAE / MAPE / accuracy / within-X%
│   ├── combiners.py           MAE-optimised convex blend of base learners
│   ├── train.py               CV, stacking, calibration, artifact export
│   ├── validate_savings.py    measures whether the booking advice holds up
│   ├── predict.py             FarePredictor inference API
│   ├── advisor.py             BookingAdvisor cost-saving recommendations
│   └── eda.py                 figures and data profile
├── tests/                     pytest suite (34 tests)
├── models/                    airfare_model.joblib + one file per base learner
└── outputs/
    ├── submission.xlsx        predictions for Test_set.xlsx
    ├── figures/               7 EDA charts
    └── reports/               cv_metrics, fare_band_metrics, savings_validation,
                               feature_importance
```

Run the tests with:

```bash
python -m pytest tests -q
```

---

## Known limits

These are the things that would bite you, stated plainly.

- **Expensive fares are the weak spot.** Ordinary tickets are predicted well; fares
  above ₹25,000 are only 82 rows in the training data and are still under-predicted
  even after calibration. Treat any estimate above ₹25,000 as indicative. Business
  class (six training rows) should not be trusted at all.
- **The model only knows one season.** Journey dates outside 1 March – 27 June 2019
  are extrapolation. `FarePredictor` raises a `RuntimeWarning` on them, exposes an
  `in_training_range` column, and the app shows a banner — but it will still return a
  number, so check the flag.
- **Selection was done on the same folds as evaluation.** Several feature sets,
  two combiners and a calibrator were all compared against the same 10-fold split, so
  the headline figure carries some selection optimism. Read it as "about 0.92", not
  as a fourth-decimal fact. Nested CV would settle it.
- **No booking date exists in the data**, so days-to-departure — in reality one of the
  strongest fare drivers — cannot be modelled at all. Day-of-year and
  distance-to-holiday are partial proxies.
- **Absolute rupee values are of their time.** What generalises is the structure the
  model learns: route, carrier, stops, timing, seasonality.
- **The advisor cannot invent flights.** It re-prices itineraries that actually appear
  in the data.
- **The artifact is large.** The two forests are roughly 90% of it. Shrinking them was
  measured rather than assumed — dropping Extra Trees from 500 trees at
  `min_samples_leaf=2` to 400 at 3 saves 25 MB but costs about 0.010 of R², so the
  accurate settings were kept and each learner is saved to its own file instead, so
  that no single file exceeds the 100 MB most git hosts reject.
