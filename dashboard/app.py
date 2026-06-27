"""
=============================================================================
 GetAround — Dashboard d'analyse des délais (INTERFACE Streamlit)
=============================================================================
Ce fichier ne contient QUE de l'affichage. Tous les calculs sont délégués au
module utils.py (séparation logique / interface).

Plan de l'écran, en 4 sections qui racontent une histoire :
   1. À quelle fréquence les voitures sont-elles rendues en retard ?
   2. Quel impact sur le conducteur suivant ?
   3. Simulateur de seuil (interactif).
   4. Lecture des données / recommandation.
=============================================================================
"""
import numpy as np
import pandas as pd
import plotly.express as px        # graphiques rapides (histogramme, barres)
import plotly.graph_objects as go  # graphiques sur mesure (courbes, barres custom)
import streamlit as st

import utils  # notre module métier

# --------------------------------------------------------------------------- #
# Configuration de la page + thème visuel
# --------------------------------------------------------------------------- #
# set_page_config DOIT être la première commande Streamlit appelée.
st.set_page_config(
    page_title="GetAround · Delay Analysis",
    page_icon="🚗",
    layout="wide",                       # pleine largeur
    initial_sidebar_state="expanded",    # barre latérale ouverte au démarrage
)

# Palette de couleurs (identité visuelle violette de GetAround) centralisée
# dans des constantes : on change la charte à un seul endroit.
PRIMARY = "#6C3FFF"   # violet principal
ACCENT = "#FF5A5F"    # rouge/corail (alertes, retards)
INK = "#1B1340"       # texte foncé
MUTED = "#8A85A8"     # gris pour texte secondaire
GOOD = "#11A66A"      # vert (problèmes résolus)

# Un peu de CSS injecté pour soigner la typographie et les "pilules"/encadrés.
# unsafe_allow_html=True est nécessaire pour que Streamlit interprète le HTML.
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


# --------------------------------------------------------------------------- #
# Chargement des données (mis en cache)
# --------------------------------------------------------------------------- #
# @st.cache_data : Streamlit ré-exécute TOUT le script à chaque interaction
# (ex. déplacement du slider). Le cache garantit que le chargement Excel et les
# calculs lourds ne se font QU'UNE SEULE FOIS, pas à chaque coup de slider.
@st.cache_data
def get_data():
    df = utils.load_rentals()                 # table brute
    chain = utils.build_chained_pairs(df)     # paires enchaînées (self-join)
    metrics = utils.headline_metrics(df, chain)
    return df, chain, metrics


df, chain, M = get_data()
TOTAL = M["total_rentals"]


# --------------------------------------------------------------------------- #
# En-tête : titre + contexte + ligne de KPI
# --------------------------------------------------------------------------- #
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

# st.columns(5) crée 5 colonnes côte à côte pour aligner 5 indicateurs.
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total rentals", f"{M['total_rentals']:,}")
k2.metric("Returned late", f"{M['late_rate']*100:.0f}%", help="Share of ended rentals checked out after the planned time")
k3.metric("Median lateness", f"{M['median_late']:.0f} min", help="Among late returns only")
k4.metric("Back-to-back rentals", f"{M['chained_share']*100:.1f}%", help="Have a previous rental of the same car within 12h — the population the feature touches")
k5.metric(
    "Late return → impact",
    f"{M['impacted_rate']*100:.1f}%",
    help="Share of back-to-back rentals where the previous car came back after the planned start",
)

st.divider()

# --------------------------------------------------------------------------- #
# SECTION 1 — Fréquence et gravité des retards
# --------------------------------------------------------------------------- #
st.header("1 · How often are cars returned late?")
c1, c2 = st.columns([3, 2])   # ratio de largeur 3:2 entre les deux graphiques

with c1:
    ended = df[df["state"] == "ended"].copy()
    d = ended["delay_at_checkout_in_minutes"].dropna()
    # IMPORTANT : on "clippe" à ±300 min UNIQUEMENT pour l'AFFICHAGE
    # (sinon les outliers extrêmes ±70 000 min écrasent l'histogramme).
    # La logique métier de simulation, elle, garde les valeurs brutes.
    d_clip = d.clip(-300, 300)
    fig = px.histogram(d_clip, nbins=60, color_discrete_sequence=[PRIMARY])
    fig.add_vline(x=0, line_dash="dash", line_color=INK)  # repère "à l'heure"
    fig.update_layout(
        title="Checkout delay distribution (clipped to ±5h)",
        xaxis_title="Delay at checkout (minutes)  ·  negative = returned early",
        yaxis_title="Rentals", showlegend=False, bargap=0.02,
        plot_bgcolor="white", height=360, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig, width='stretch')

with c2:
    # Parmi les retards positifs, on range en tranches de gravité.
    late = ended.loc[ended["delay_at_checkout_in_minutes"] > 0, "delay_at_checkout_in_minutes"]
    buckets = pd.cut(
        late, [0, 15, 30, 60, 120, np.inf],
        labels=["0–15 min", "15–30 min", "30–60 min", "1–2 h", "2 h +"],
    ).value_counts().sort_index()
    fig2 = px.bar(
        x=buckets.values, y=buckets.index, orientation="h",
        color_discrete_sequence=[ACCENT],
    )
    fig2.update_layout(
        title="How late, when late",
        xaxis_title="Rentals", yaxis_title="",
        plot_bgcolor="white", height=360, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig2, width='stretch')

st.markdown(
    f'<p class="lead"><b>{M["late_rate"]*100:.0f}% of returns are late</b>, with a '
    f"median lateness of {M['median_late']:.0f} min — but over a quarter of late "
    "returns exceed 2 hours, which is what hurts the next driver.</p>",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------------------------------- #
# SECTION 2 — Impact sur le conducteur suivant
# --------------------------------------------------------------------------- #
st.header("2 · What does it do to the next driver?")
c3, c4 = st.columns(2)

with c3:
    # Taux d'impact ventilé par type de check-in (mobile vs connect).
    known = chain.dropna(subset=["previous_delay_at_checkout"])
    by_type = known.groupby("checkin_type")["impacted"].mean().mul(100)
    fig3 = px.bar(
        x=by_type.index, y=by_type.values,
        color=by_type.index,
        color_discrete_map={"mobile": ACCENT, "connect": PRIMARY},
    )
    fig3.update_layout(
        title="Share of back-to-back rentals impacted, by check-in type",
        xaxis_title="", yaxis_title="% impacted", showlegend=False,
        plot_bgcolor="white", height=340, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig3, width='stretch')

with c4:
    # Comparaison du taux d'annulation : non-impactées vs impactées.
    # cancel_lift = (taux si impactée, taux sinon) calculé dans utils.
    imp, clean = M["cancel_lift"]
    fig4 = go.Figure(go.Bar(
        x=["Not impacted", "Impacted by late return"],
        y=[clean * 100, imp * 100],
        marker_color=[MUTED, ACCENT],
        text=[f"{clean*100:.0f}%", f"{imp*100:.0f}%"], textposition="outside",
    ))
    fig4.update_layout(
        title="Cancellation rate rises when the previous car is late",
        yaxis_title="Cancellation rate", plot_bgcolor="white",
        height=340, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig4, width='stretch')

st.markdown(
    '<p class="lead">Late returns hit <b>mobile</b> check-ins almost twice as '
    "often as <b>Connect</b>, and an impacted rental is markedly more likely to "
    "be canceled. That is the harm a minimum-delay buffer is meant to prevent.</p>",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------------------------------- #
# SECTION 3 — Simulateur de seuil (interactif)
# --------------------------------------------------------------------------- #
st.header("3 · Simulate a threshold")

# Les contrôles sont placés dans la barre latérale (st.sidebar).
# À chaque changement, Streamlit relance le script -> les KPI/graphes se
# recalculent (mais les données restent en cache, donc c'est instantané).
with st.sidebar:
    st.markdown("### ⚙️ Simulation")
    scope = st.radio(
        "Scope", ["all", "connect"],
        format_func=lambda s: "All cars" if s == "all" else "Connect only",
    )
    threshold = st.slider("Minimum delay between rentals (minutes)", 0, 720, 120, 15)
    st.caption(
        "A booking is **blocked** if its planned gap with the previous rental "
        "is shorter than the threshold. A problem is **solved** if the buffer "
        "now covers the previous driver's lateness."
    )

# Un seul appel pour obtenir tous les chiffres de la configuration choisie.
sim = utils.simulate(chain, threshold, scope, TOTAL)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rentals blocked", f"{sim['blocked']:,}", help="Bookings prevented because the gap is shorter than the threshold")
m2.metric("…as % of all rentals", f"{sim['blocked_share_total']*100:.2f}%")
m3.metric("Problem cases solved", f"{sim['solved']} / {sim['impacted']}")
m4.metric("…% of problems solved", f"{sim['solved_share']*100:.0f}%")

# Courbe d'arbitrage : on superpose les deux séries (résolus vs bloquées)
# sur tout le balayage de seuils, et on marque le seuil courant en pointillé.
curve = utils.tradeoff_curve(chain, scope, TOTAL)
fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=curve["threshold"], y=curve["pct_problems_solved"],
    name="% problems solved", line=dict(color=GOOD, width=3)))
fig5.add_trace(go.Scatter(
    x=curve["threshold"], y=curve["pct_rentals_blocked"],
    name="% of all rentals blocked", line=dict(color=ACCENT, width=3)))
fig5.add_vline(x=threshold, line_dash="dash", line_color=INK)  # position du curseur
fig5.update_layout(
    title=f"Trade-off curve — scope: {'all cars' if scope=='all' else 'Connect only'}",
    xaxis_title="Threshold (minutes)", yaxis_title="Percent",
    plot_bgcolor="white", height=420, margin=dict(t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig5, width='stretch')

# Phrase de synthèse dynamique : se met à jour avec le seuil/périmètre choisi.
st.markdown(
    f'<p class="lead">At <b>{threshold} min</b> on <b>'
    f'{"all cars" if scope=="all" else "Connect only"}</b>, you solve '
    f'<b>{sim["solved_share"]*100:.0f}%</b> of problematic hand-overs while '
    f'blocking <b>{sim["blocked_share_total"]*100:.2f}%</b> of all rentals '
    f'({sim["blocked"]:,} bookings).</p>',
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------------------------------- #
# SECTION 4 — Lecture des données / recommandation
# --------------------------------------------------------------------------- #
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
