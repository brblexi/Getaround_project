"""GetAround — delay analysis (business logic).

Kept separate from `app.py`, which only renders. The split means these
functions can be unit-tested without starting Streamlit and re-used from a
notebook.

Vocabulary
----------
back-to-back pair
    A rental that follows another rental of the *same* car. The source data
    only fills `previous_ended_rental_id` when the previous rental ended within
    12 hours, so the pairing is already bounded.
gap (g)
    `time_delta_with_previous_rental_in_minutes` — the *planned* interval
    between the end of the previous rental and the start of this one.
delay (d)
    The checkout delay of the *previous* rental, i.e. how late that car came
    back.
impacted
    `d > g`: the previous car was returned after this rental's planned start,
    so the next driver waits.

Threshold logic
---------------
Enforcing a minimum delay `T` makes the effective interval `max(g, T)`. A case
is still a problem if `d > max(g, T)`, so a currently problematic case is
*solved* when `d <= T`. That gives the three quantities the simulator reports:
blocked (`g < T`, the cost), impacted (`d > g`, the problem) and solved
(impacted and `d <= T`, the benefit).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Relative to this module, so the path holds locally, in the Docker image and on
# the Space. Overridable for tests and for pointing at a refreshed extract.
DATA_PATH = Path(
    os.getenv("RENTALS_PATH", Path(__file__).parent / "data" / "get_around_delay_analysis.xlsx")
)

LATENESS_BINS = [0, 15, 30, 60, 120, np.inf]
LATENESS_LABELS = ["0–15 min", "15–30 min", "30–60 min", "1–2 h", "2 h +"]


def load_rentals(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw `rentals_data` sheet (21,310 rows)."""
    return pd.read_excel(path, sheet_name="rentals_data")


def build_chained_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Join each rental to the checkout delay of the rental that preceded it.

    Every row carries the id of its previous rental, but that rental's delay
    lives in a different row — so the table is joined to itself on
    `previous_ended_rental_id`.

    The join is inner, which keeps only rentals that have a predecessor. It is
    safe against pandas matching NaN keys to each other, because the right-hand
    key is built from `rental_id`, which is never null.

    Rows with no planned gap are dropped: without `g` there is nothing to
    simulate. A null previous delay is kept — it means the delay is unknown,
    not that the car came back on time — and the comparisons below treat it as
    not impacted rather than guessing.
    """
    previous = df[["rental_id", "delay_at_checkout_in_minutes"]].rename(
        columns={
            "rental_id": "previous_ended_rental_id",  # join key
            "delay_at_checkout_in_minutes": "previous_delay_at_checkout",
        }
    )

    chain = df.merge(previous, on="previous_ended_rental_id", how="inner")
    chain = chain.dropna(subset=["time_delta_with_previous_rental_in_minutes"]).copy()

    # How far the previous delay eats into the available gap.
    # overlap > 0  <=>  d > g  <=>  this rental is impacted.
    chain["overlap_minutes"] = (
        chain["previous_delay_at_checkout"]
        - chain["time_delta_with_previous_rental_in_minutes"]
    )
    chain["impacted"] = chain["overlap_minutes"] > 0
    return chain


def headline_metrics(df: pd.DataFrame, chain: pd.DataFrame) -> dict:
    """Compute the KPIs shown at the top of the dashboard."""
    ended = df[df["state"] == "ended"]
    known_delay = ended["delay_at_checkout_in_minutes"].notna()
    # Late means strictly positive; a negative delay is an early return.
    late = ended.loc[known_delay & (ended["delay_at_checkout_in_minutes"] > 0)]

    # Cancellation rate, impacted against not impacted, restricted to pairs
    # whose previous delay is known.
    measured = chain.dropna(subset=["previous_delay_at_checkout"])
    cancel_impacted = measured.loc[measured["impacted"], "state"].eq("canceled").mean()
    cancel_clean = measured.loc[~measured["impacted"], "state"].eq("canceled").mean()

    return {
        "total_rentals": len(df),
        "cancel_rate": (df["state"] == "canceled").mean(),
        "late_rate": len(late) / known_delay.sum(),
        "median_late": late["delay_at_checkout_in_minutes"].median(),
        # Measured on the raw column rather than on `chain`, which has already
        # dropped pairs with no planned gap: this is the share of rentals the
        # feature could touch at all, ~8.6%, not the share we can simulate.
        "chained_share": df["previous_ended_rental_id"].notna().mean(),
        "simulable_pairs": len(chain),
        "impacted_rate": measured["impacted"].mean(),
        # (rate when impacted, rate otherwise) — about 0.17 against 0.11.
        "cancel_lift": (cancel_impacted, cancel_clean),
    }


def simulate(chain: pd.DataFrame, threshold: int, scope: str, total_rentals: int) -> dict:
    """Effect of a minimum-delay `threshold` (minutes) over a given `scope`.

    `scope` is either "all" or "connect". Shares of blocked rentals are
    expressed against *all* rentals, not against the scope, so that a
    Connect-only rule and a fleet-wide rule stay comparable.
    """
    subset = chain if scope == "all" else chain[chain["checkin_type"] == "connect"]
    gap = subset["time_delta_with_previous_rental_in_minutes"]
    delay = subset["previous_delay_at_checkout"]  # may be NaN

    blocked = int((gap < threshold).sum())
    # NaN comparisons yield False, so pairs with an unknown previous delay are
    # counted as not impacted — the conservative reading.
    impacted_mask = delay > gap
    impacted = int(impacted_mask.sum())
    solved = int((impacted_mask & (delay <= threshold)).sum())

    return {
        "blocked": blocked,
        "blocked_share_total": blocked / total_rentals,
        "blocked_share_scope": blocked / len(subset) if len(subset) else 0.0,
        "impacted": impacted,
        "solved": solved,
        "remaining": impacted - solved,
        "solved_share": solved / impacted if impacted else 0.0,
        "scope_size": len(subset),
    }


def tradeoff_curve(
    chain: pd.DataFrame,
    scope: str,
    total_rentals: int,
    thresholds=range(0, 721, 15),
) -> pd.DataFrame:
    """Sweep every threshold and return blocked rentals against solved problems.

    This is what the dashboard's trade-off curve plots. The upper bound of 720
    minutes matches the 12-hour window beyond which the source data stops
    linking a rental to its predecessor.
    """
    return pd.DataFrame(
        [
            {
                "threshold": t,
                "rentals_blocked": s["blocked"],
                "pct_rentals_blocked": s["blocked_share_total"] * 100,
                "problems_solved": s["solved"],
                "pct_problems_solved": s["solved_share"] * 100,
            }
            for t, s in ((t, simulate(chain, t, scope, total_rentals)) for t in thresholds)
        ]
    )


def checkout_delays(rentals: pd.DataFrame) -> pd.Series:
    """Checkout delays in minutes for completed rentals, nulls dropped.

    Raw values: clipping is a plotting concern and belongs to the caller, never
    to the numbers the simulation runs on.
    """
    ended = rentals.loc[rentals["state"] == "ended"]
    return ended["delay_at_checkout_in_minutes"].dropna()


def lateness_buckets(rentals: pd.DataFrame) -> pd.Series:
    """Count late returns by severity band, mildest first."""
    delays = checkout_delays(rentals)
    late = delays[delays > 0]
    return pd.cut(late, LATENESS_BINS, labels=LATENESS_LABELS).value_counts().sort_index()


def impact_rate_by_checkin(chained: pd.DataFrame) -> pd.Series:
    """Share of back-to-back rentals impacted by a late return, per check-in type.

    Pairs whose previous delay is unknown are excluded: they are missing
    measurements, not evidence of an on-time return.
    """
    measured = chained.dropna(subset=["previous_delay_at_checkout"])
    return measured.groupby("checkin_type")["impacted"].mean()
