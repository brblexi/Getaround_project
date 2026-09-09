"""Smoke tests for the Streamlit dashboard.

Run from `dashboard/`:  pytest -q

`test_utils.py` checks that the numbers are right. This file checks that the
app renders them: that the script runs top to bottom without raising, that the
widgets are wired, and that moving them re-runs cleanly. It is deliberately
shallow — an app that renders is a low bar, but it is the bar that catches a
typo in a f-string or a column renamed in utils, and it catches it before a
deploy rather than after.

AppTest runs the script in-process, without a browser. Plotly charts are not
exposed by the harness, so the assertions below target the metrics and the
sidebar widgets; a broken chart still surfaces here as an exception.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import utils

APP = str(Path(__file__).parent / "app.py")

# Reading the Excel file and building the pairs takes a few seconds on the
# first run; the default 3 second timeout is too tight.
TIMEOUT = 60


pytestmark = pytest.mark.skipif(
    not utils.DATA_PATH.exists(),
    reason=f"dataset not found at {utils.DATA_PATH} — check that Git LFS files were pulled",
)


@pytest.fixture(scope="module")
def app() -> AppTest:
    """Run the app once and reuse it: the module-scoped cache makes the
    re-runs in the tests below near-instant."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    return at


def metric(at: AppTest, label: str):
    """Find a metric by its label rather than by position.

    Positional indexing would break the moment a KPI is inserted, and the
    failure would point at the wrong test.
    """
    return next(m for m in at.metric if m.label == label)


def test_the_app_renders_without_raising(app):
    assert not app.exception


def test_the_headline_kpis_are_present(app):
    labels = {m.label for m in app.metric}
    assert {
        "Total rentals",
        "Returned late",
        "Median lateness",
        "Back-to-back rentals",
        "Rentals blocked",
    } <= labels


def test_the_four_sections_are_rendered(app):
    headers = " ".join(h.value for h in app.header)
    for section in ["1 ·", "2 ·", "3 ·", "4 ·"]:
        assert section in headers


def test_the_simulator_controls_are_in_the_sidebar(app):
    assert len(app.sidebar.radio) == 1
    assert len(app.sidebar.slider) == 1
    assert app.sidebar.slider[0].value == 120  # the recommended default


def test_a_zero_threshold_blocks_nothing():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.sidebar.slider[0].set_value(0).run()

    assert not at.exception
    assert metric(at, "Rentals blocked").value == "0"


def test_raising_the_threshold_blocks_more_rentals():
    """The trade-off the whole page exists to show: more buffer, more blocked
    bookings. If this ever inverts, the recommendation is wrong."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()

    at.sidebar.slider[0].set_value(60).run()
    few = int(metric(at, "Rentals blocked").value.replace(",", ""))

    at.sidebar.slider[0].set_value(360).run()
    many = int(metric(at, "Rentals blocked").value.replace(",", ""))

    assert 0 < few < many


def test_switching_to_connect_only_narrows_the_scope():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()

    at.sidebar.slider[0].set_value(360).run()
    all_cars = int(metric(at, "Rentals blocked").value.replace(",", ""))

    at.sidebar.radio[0].set_value("connect").run()
    connect_only = int(metric(at, "Rentals blocked").value.replace(",", ""))

    assert not at.exception
    assert connect_only < all_cars
