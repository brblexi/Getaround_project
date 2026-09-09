"""GetAround — delay analysis dashboard (presentation layer).

This module only renders: every computation lives in `utils.py`, which keeps the
business logic importable and testable outside Streamlit.

Streamlit re-runs this entire script top to bottom on every interaction — there
is no event loop and no callback. That is why the file is plain module-level
code, and why anything expensive goes behind @st.cache_data.

Layout, in four sections:
    1. How often are cars returned late?
    2. What does a late return do to the next driver?
    3. Threshold simulator.
    4. Reading of the data and recommendation.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

# set_page_config must be the first Streamlit call.
st.set_page_config(
    page_title="GetAround · Delay Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand palette, centralised so the theme changes in one place.
PRIMARY = "#6C3FFF"  # violet
ACCENT = "#FF5A5F"   # coral: lateness, alerts
INK = "#1B1340"      # headings and body text
MUTED = "#8A85A8"    # secondary text
GOOD = "#11A66A"     # solved problems

# Display-only bound for the delay histogram: raw values reach ±70,000 minutes,
# which would flatten the entire distribution. Simulations use raw values.
DELAY_CLIP_MINUTES = 300


# Streamlit exposes no API for content width or custom components, so the pill
# and the recommendation box need raw CSS. Nothing here comes from user input,
# so the "unsafe" flag carries no injection risk; the fragile part is depending
# on .block-container, an internal class Streamlit could rename.
st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 2.2rem; max-width: 1180px;}}
      h1, h2, h3 {{color: {INK}; font-weight: 700;}}
      .lead {{color: {MUTED}; font-size: 1.02rem; line-height: 1.5;}}
      div[data-testid="stMetricValue"] {{color: {INK}; font-weight: 700;}}
      .pill {{display:inline-block; padding:2px 10px; border-radius:999px;
              background:{PRIMARY}1A; color:{PRIMARY}; font-size:.8rem;
              font-weight:600; margin-bottom:.4rem;}}
      .reco {{background:{PRIMARY}0D; border-left:4px solid {PRIMARY};
              padding:1rem 1.2rem; border-radius:8px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    """Load the dataset and derive the aggregates used across the page.

    Streamlit re-runs the whole script on every interaction; caching keeps the
    Excel read and the self-join out of the slider's critical path.
    """
    rentals = utils.load_rentals()
    chained = utils.build_chained_pairs(rentals)
    return rentals, chained, utils.headline_metrics(rentals, chained)


df, chain, metrics = get_data()
TOTAL_RENTALS = metrics["total_rentals"]


@st.cache_data
def get_curve(_chain, scope, total_rentals):
    """Cache the trade-off curve, which depends only on the scope.

    The curve is 49 simulations over the whole threshold range; without this it
    would be recomputed on every slider move to redraw an identical line.
    The leading underscore tells Streamlit not to hash the DataFrame — it
    cannot — while still keying the cache on scope.
    """
    return utils.tradeoff_curve(_chain, scope, total_rentals)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<span class="pill">Product analytics</span>', unsafe_allow_html=True)
st.title("GetAround — Should we enforce a delay between rentals?")
st.markdown(
    '<p class="lead">When a driver returns a car late, it eats into the next '
    "rental of the same car — causing friction and cancellations. The Product "
    "team is considering a <b>minimum delay between two rentals</b>. This "
    "dashboard quantifies the problem and lets you simulate a threshold and its "
    "scope to balance <b>fewer problems</b> against <b>lost availability</b>.</p>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total rentals", f"{metrics['total_rentals']:,}")
k2.metric(
    "Returned late",
    f"{metrics['late_rate'] * 100:.0f}%",
    help="Share of ended rentals checked out after the planned time",
)
k3.metric(
    "Median lateness",
    f"{metrics['median_late']:.0f} min",
    help="Among late returns only",
)
k4.metric(
    "Back-to-back rentals",
    f"{metrics['chained_share'] * 100:.1f}%",
    help="Rentals preceded by another rental of the same car within 12h — the population the feature touches",
)
k5.metric(
    "Late return → impact",
    f"{metrics['impacted_rate'] * 100:.1f}%",
    help="Share of back-to-back rentals where the previous car came back after the planned start",
)

st.divider()

# ---------------------------------------------------------------------------
# 1 — Frequency and severity of late returns
# ---------------------------------------------------------------------------
st.header("1 · How often are cars returned late?")
left, right = st.columns([3, 2])

with left:
    delays = utils.checkout_delays(df)
    # clip() bends the picture, not the data: outliers pile onto the bounds so
    # the bulk of the distribution stays readable. utils returns raw values and
    # simulate() never sees a clipped number.
    fig = px.histogram(
        delays.clip(-DELAY_CLIP_MINUTES, DELAY_CLIP_MINUTES),
        nbins=60,
        color_discrete_sequence=[PRIMARY],
    )
    fig.add_vline(x=0, line_dash="dash", line_color=INK)  # on-time reference
    fig.update_layout(
        title=f"Checkout delay distribution (clipped to ±{DELAY_CLIP_MINUTES // 60}h)",
        xaxis_title="Delay at checkout (minutes)  ·  negative = returned early",
        yaxis_title="Rentals",
        showlegend=False,
        bargap=0.02,
        plot_bgcolor="white",
        height=360,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")

with right:
    buckets = utils.lateness_buckets(df)
    fig2 = px.bar(
        x=buckets.values,
        y=buckets.index,
        orientation="h",
        color_discrete_sequence=[ACCENT],
    )
    fig2.update_layout(
        title="How late, when late",
        xaxis_title="Rentals",
        yaxis_title="",
        plot_bgcolor="white",
        height=360,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig2, width="stretch")

severe_share = buckets["2 h +"] / buckets.sum()
st.markdown(
    f'<p class="lead"><b>{metrics["late_rate"] * 100:.0f}% of returns are late</b>, with a '
    f"median lateness of {metrics['median_late']:.0f} min — but "
    f"{severe_share * 100:.0f}% of late returns exceed two hours, which is what "
    "hurts the next driver.</p>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# 2 — Impact on the next driver
# ---------------------------------------------------------------------------
st.header("2 · What does it do to the next driver?")
left, right = st.columns(2)

with left:
    by_type = utils.impact_rate_by_checkin(chain)
    fig3 = px.bar(
        x=by_type.index,
        y=by_type.values * 100,
        color=by_type.index,
        color_discrete_map={"mobile": ACCENT, "connect": PRIMARY},
    )
    fig3.update_layout(
        title="Share of back-to-back rentals impacted, by check-in type",
        xaxis_title="",
        yaxis_title="% impacted",
        showlegend=False,
        plot_bgcolor="white",
        height=340,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig3, width="stretch")

with right:
    impacted_rate, clean_rate = metrics["cancel_lift"]
    fig4 = go.Figure(
        go.Bar(
            x=["Not impacted", "Impacted by late return"],
            y=[clean_rate * 100, impacted_rate * 100],
            marker_color=[MUTED, ACCENT],
            text=[f"{clean_rate * 100:.0f}%", f"{impacted_rate * 100:.0f}%"],
            textposition="outside",
        )
    )
    fig4.update_layout(
        title="Cancellation rate rises when the previous car is late",
        yaxis_title="Cancellation rate",
        plot_bgcolor="white",
        height=340,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig4, width="stretch")

st.markdown(
    '<p class="lead">Late returns hit <b>mobile</b> check-ins almost twice as '
    "often as <b>Connect</b>, and an impacted rental is markedly more likely to "
    "be canceled. That is the harm a minimum-delay buffer is meant to prevent.</p>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# 3 — Threshold simulator
# ---------------------------------------------------------------------------
st.header("3 · Simulate a threshold")

with st.sidebar:
    st.markdown("### ⚙️ Simulation")
    scope = st.radio(
        "Scope",
        ["all", "connect"],
        format_func=lambda s: "All cars" if s == "all" else "Connect only",
    )
    threshold = st.slider("Minimum delay between rentals (minutes)", 0, 720, 120, 15)
    st.caption(
        "A booking is **blocked** if its planned gap with the previous rental "
        "is shorter than the threshold. A problem is **solved** if the buffer "
        "now covers the previous driver's lateness."
    )

sim = utils.simulate(chain, threshold, scope, TOTAL_RENTALS)

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Rentals blocked",
    f"{sim['blocked']:,}",
    help="Bookings prevented because the gap is shorter than the threshold",
)
m2.metric("…as % of all rentals", f"{sim['blocked_share_total'] * 100:.2f}%")
m3.metric("Problem cases solved", f"{sim['solved']} / {sim['impacted']}")
m4.metric("…% of problems solved", f"{sim['solved_share'] * 100:.0f}%")

# Trade-off curve: both series swept over the full threshold range, with the
# current slider position marked.
curve = get_curve(chain, scope, TOTAL_RENTALS)
fig5 = go.Figure()
fig5.add_trace(
    go.Scatter(
        x=curve["threshold"],
        y=curve["pct_problems_solved"],
        name="% problems solved",
        line=dict(color=GOOD, width=3),
    )
)
fig5.add_trace(
    go.Scatter(
        x=curve["threshold"],
        y=curve["pct_rentals_blocked"],
        name="% of all rentals blocked",
        line=dict(color=ACCENT, width=3),
    )
)
fig5.add_vline(x=threshold, line_dash="dash", line_color=INK)
fig5.update_layout(
    title=f"Trade-off curve — scope: {'all cars' if scope == 'all' else 'Connect only'}",
    xaxis_title="Threshold (minutes)",
    yaxis_title="Percent",
    plot_bgcolor="white",
    height=420,
    margin=dict(t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig5, width="stretch")

st.markdown(
    f'<p class="lead">At <b>{threshold} min</b> on <b>'
    f'{"all cars" if scope == "all" else "Connect only"}</b>, you solve '
    f'<b>{sim["solved_share"] * 100:.0f}%</b> of problematic hand-overs while '
    f'blocking <b>{sim["blocked_share_total"] * 100:.2f}%</b> of all rentals '
    f'({sim["blocked"]:,} bookings).</p>',
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# 4 — Reading of the data
# ---------------------------------------------------------------------------
st.header("4 · Reading of the data")
st.markdown(
    """
<div class="reco">
<b>The lever is real but narrow.</b> Only ~9% of rentals are back-to-back, so the
feature touches a small slice — but within it, late returns clearly raise
cancellations.<br><br>
<b>Returns diminish fast.</b> Pushing the threshold up keeps solving more problems,
but each extra block of minutes buys fewer solved cases while it keeps removing
available bookings — the curves cross into diminishing territory past ~2 hours.<br><br>
<b>Scope matters more than width.</b> Lateness concentrates on <i>mobile</i>
check-ins, so a <i>Connect-only</i> rule blocks far fewer bookings but also leaves
most problem cases (which are mobile) untouched. A threshold around
<b>60–120 min on all cars</b> captures roughly half to two-thirds of problems
while blocking under ~3% of rentals — a defensible starting point to A/B test.
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Source: get_around_delay_analysis.xlsx · 'impacted' = previous car returned "
    "after this rental's planned start · 'solved' = buffer ≥ previous lateness."
)
