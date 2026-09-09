"""Tests for the delay-analysis logic.

Run from `dashboard/`:  pytest -q

The fixture is a hand-built table of six rentals, small enough that every
expected number below can be checked by reading it. Testing against the real
Excel file would only tell us the file has not changed; testing against a table
we control tells us the *rules* are right — including the edge cases that
barely occur in the real data.
"""

import numpy as np
import pandas as pd
import pytest

import utils


@pytest.fixture
def rentals() -> pd.DataFrame:
    """Six rentals: three of them follow another rental of the same car.

    id 3 follows 1: previous car 90 min late, 60 min gap  -> impacted (30 over)
    id 4 follows 2: previous car 10 min late, 120 min gap -> not impacted
    id 5 follows 6: previous delay unknown (cancelled)    -> not impacted
    """
    return pd.DataFrame(
        [
            # id, state,      checkin,   delay, previous_id, gap
            (1, "ended", "mobile", 90.0, np.nan, np.nan),
            (2, "ended", "connect", 10.0, np.nan, np.nan),
            (3, "canceled", "mobile", np.nan, 1.0, 60.0),
            (4, "ended", "connect", -15.0, 2.0, 120.0),
            (5, "ended", "mobile", 5.0, 6.0, 30.0),
            (6, "canceled", "mobile", np.nan, np.nan, np.nan),
        ],
        columns=[
            "rental_id",
            "state",
            "checkin_type",
            "delay_at_checkout_in_minutes",
            "previous_ended_rental_id",
            "time_delta_with_previous_rental_in_minutes",
        ],
    )


@pytest.fixture
def chain(rentals) -> pd.DataFrame:
    return utils.build_chained_pairs(rentals)


# --------------------------------------------------------------------------
# build_chained_pairs
# --------------------------------------------------------------------------

def test_self_join_attaches_the_previous_rentals_delay(chain):
    row = chain.set_index("rental_id").loc[3]
    assert row["previous_delay_at_checkout"] == 90.0
    assert row["overlap_minutes"] == 30.0  # 90 late against a 60 min gap
    assert row["impacted"]


def test_rentals_without_a_predecessor_are_excluded(chain):
    assert set(chain["rental_id"]) == {3, 4, 5}


def test_an_unknown_previous_delay_is_not_counted_as_impacted(chain):
    # Rental 5 follows a cancelled rental, so its delay is NaN. NaN > gap is
    # False, which is the conservative reading: unknown is not evidence of harm.
    assert not chain.set_index("rental_id").loc[5, "impacted"]


def test_null_keys_do_not_join_to_each_other(rentals):
    """pandas will happily match NaN to NaN on a merge key.

    Here the right-hand key comes from `rental_id`, which is never null, so the
    four rentals with no predecessor cannot pair up with each other. This test
    fails loudly if that ever changes.
    """
    assert utils.build_chained_pairs(rentals)["previous_ended_rental_id"].notna().all()


# --------------------------------------------------------------------------
# simulate
# --------------------------------------------------------------------------

def test_a_zero_threshold_changes_nothing(chain):
    result = utils.simulate(chain, threshold=0, scope="all", total_rentals=6)
    assert result["blocked"] == 0
    assert result["solved"] == 0
    assert result["impacted"] == 1  # only rental 3


def test_a_threshold_above_the_delay_solves_the_case(chain):
    # The previous car was 90 min late, so a 120 min buffer covers it.
    result = utils.simulate(chain, threshold=120, scope="all", total_rentals=6)
    assert result["solved"] == 1
    assert result["remaining"] == 0
    assert result["solved_share"] == 1.0


def test_a_threshold_below_the_delay_does_not(chain):
    result = utils.simulate(chain, threshold=60, scope="all", total_rentals=6)
    assert result["solved"] == 0
    assert result["remaining"] == 1


def test_blocked_counts_gaps_shorter_than_the_threshold(chain):
    # Gaps are 60, 120 and 30 minutes; a 90 min buffer blocks two of them.
    result = utils.simulate(chain, threshold=90, scope="all", total_rentals=6)
    assert result["blocked"] == 2
    assert result["blocked_share_total"] == pytest.approx(2 / 6)


def test_scope_restricts_to_connect_rentals(chain):
    result = utils.simulate(chain, threshold=720, scope="connect", total_rentals=6)
    assert result["scope_size"] == 1  # only rental 4
    assert result["impacted"] == 0  # the impacted one is a mobile check-in


def test_blocked_share_is_always_expressed_against_all_rentals(chain):
    """Otherwise a Connect-only rule looks worse than it is: dividing by a
    smaller scope inflates the percentage and the two options stop being
    comparable, which is the whole point of the simulator."""
    result = utils.simulate(chain, threshold=720, scope="connect", total_rentals=6)
    assert result["blocked_share_total"] == pytest.approx(result["blocked"] / 6)


# --------------------------------------------------------------------------
# tradeoff_curve
# --------------------------------------------------------------------------

def test_the_curve_is_monotonic_in_both_directions(chain):
    """Raising the threshold can only block more and solve more.

    Not obvious from the formulas, and the property a product manager relies on
    when reading the chart — if it ever breaks, the recommendation breaks too.
    """
    curve = utils.tradeoff_curve(chain, scope="all", total_rentals=6)
    assert curve["rentals_blocked"].is_monotonic_increasing
    assert curve["problems_solved"].is_monotonic_increasing


def test_the_curve_covers_the_twelve_hour_window(chain):
    curve = utils.tradeoff_curve(chain, scope="all", total_rentals=6)
    assert curve["threshold"].min() == 0
    assert curve["threshold"].max() == 720


# --------------------------------------------------------------------------
# headline_metrics and display helpers
# --------------------------------------------------------------------------

def test_headline_metrics(rentals, chain):
    metrics = utils.headline_metrics(rentals, chain)

    assert metrics["total_rentals"] == 6
    # Four rentals ended with a known delay (90, 10, -15, 5); three were late.
    assert metrics["late_rate"] == pytest.approx(0.75)
    assert metrics["median_late"] == pytest.approx(10.0)  # median of 90, 10, 5
    # Three rentals have a predecessor, out of six.
    assert metrics["chained_share"] == pytest.approx(0.5)


def test_early_returns_are_not_counted_as_late(rentals):
    """Rental 4 came back 15 minutes early. A sign error here would turn every
    early return into a late one and roughly double the headline figure."""
    delays = utils.checkout_delays(rentals)
    assert (delays < 0).sum() == 1
    assert utils.lateness_buckets(rentals).sum() == 3  # the three late ones only


def test_lateness_buckets_are_ordered_and_complete(rentals):
    buckets = utils.lateness_buckets(rentals)
    assert list(buckets.index) == utils.LATENESS_LABELS
    assert buckets["0–15 min"] == 2  # the 5 and 10 min returns
    assert buckets["1–2 h"] == 1  # the 90 min one


def test_impact_rate_by_checkin_ignores_unknown_delays(chain):
    rates = utils.impact_rate_by_checkin(chain)
    # Rental 5's previous delay is unknown, so mobile is 1 impacted out of 1
    # measured, not out of 2.
    assert rates["mobile"] == pytest.approx(1.0)
    assert rates["connect"] == pytest.approx(0.0)
