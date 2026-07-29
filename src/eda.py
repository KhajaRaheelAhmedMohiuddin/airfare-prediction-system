"""Exploratory analysis: writes the figures and the data-profile report."""

from __future__ import annotations

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURE_DIR, REPORT_DIR, TARGET
from data_prep import load_raw, prepare
from features import add_group_statistics, build_features

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sns.set_theme(style="whitegrid", palette="deep")
PALETTE = "viridis"


def save(fig, name: str) -> None:
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> None:
    raw_train, raw_test, _ = load_raw()
    train, test = prepare(drop_duplicates=True)
    feats = build_features(train)
    feats, _ = add_group_statistics(feats, build_features(test))
    feats[TARGET] = train[TARGET].to_numpy()

    print("Generating figures...")

    # 1. Fare distribution --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(feats[TARGET], bins=60, kde=True, ax=axes[0], color="#2a6f97")
    axes[0].set_title("Fare distribution (right-skewed)")
    axes[0].set_xlabel("Price (Rs.)")
    sns.histplot(np.log1p(feats[TARGET]), bins=60, kde=True, ax=axes[1], color="#468faf")
    axes[1].set_title("log1p(Fare) - the modelling target")
    axes[1].set_xlabel("log1p(Price)")
    save(fig, "01_price_distribution.png")

    # 2. Airline ------------------------------------------------------------
    order = feats.groupby("Airline")[TARGET].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=feats, x="Airline", y=TARGET, order=order, ax=ax, palette=PALETTE)
    ax.set_title("Fare by airline")
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    save(fig, "02_price_by_airline.png")

    # 3. Stops and route/day ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.boxplot(data=feats, x="Total_Stops", y=TARGET, ax=axes[0], palette=PALETTE)
    axes[0].set_title("Fare by number of stops")
    sns.boxplot(data=feats, x="Journey_Month", y=TARGET, ax=axes[1], palette=PALETTE)
    axes[1].set_title("Fare by month of travel")
    sns.boxplot(data=feats, x="Journey_DayOfWeek", y=TARGET, ax=axes[2], palette=PALETTE)
    axes[2].set_title("Fare by weekday (0=Mon)")
    save(fig, "03_stops_month_weekday.png")

    # 4. Duration vs price --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=feats.sample(min(4000, len(feats)), random_state=0),
        x="Duration_Minutes",
        y=TARGET,
        hue="Total_Stops",
        palette=PALETTE,
        alpha=0.55,
        ax=ax,
    )
    ax.set_title("Duration vs fare, coloured by stops")
    ax.set_xlabel("Duration (minutes)")
    save(fig, "04_duration_vs_price.png")

    # 5. Route heat map -----------------------------------------------------
    pivot = feats.pivot_table(
        index="Source", columns="Destination", values=TARGET, aggfunc="median"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("Median fare by city pair (Rs.)")
    save(fig, "05_route_heatmap.png")

    # 6. Departure hour -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    hourly = feats.groupby("Dep_Hour")[TARGET].agg(["median", "count"])
    ax.bar(hourly.index, hourly["median"], color="#2a6f97")
    ax.set_title("Median fare by departure hour")
    ax.set_xlabel("Departure hour")
    ax.set_ylabel("Median price (Rs.)")
    save(fig, "06_price_by_departure_hour.png")

    # 7. Numeric correlation ------------------------------------------------
    numeric = feats.select_dtypes(include=[np.number])
    corr = numeric.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)[:20]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(x=corr.values, y=corr.index, ax=ax, palette="coolwarm")
    ax.set_title("Top 20 numeric features by correlation with fare")
    save(fig, "07_feature_correlation.png")

    # ------------------------------------------------------------- profile
    profile = {
        "train_rows_raw": len(raw_train),
        "train_rows_after_dedup": len(train),
        "duplicate_rows_removed": len(raw_train) - len(train),
        "test_rows": len(raw_test),
        "engineered_features": feats.shape[1] - 1,
        "price_min": float(feats[TARGET].min()),
        "price_max": float(feats[TARGET].max()),
        "price_mean": float(feats[TARGET].mean()),
        "price_median": float(feats[TARGET].median()),
        "airlines": int(feats["Airline"].nunique()),
        "routes": int(feats["Route"].nunique()),
        "date_range": f"{raw_train['Date_of_Journey'].min()} .. {raw_train['Date_of_Journey'].max()}",
    }
    pd.Series(profile).to_csv(REPORT_DIR / "data_profile.csv", header=False)
    print("\nData profile:")
    for k, v in profile.items():
        print(f"  {k:<26} {v}")


if __name__ == "__main__":
    main()
