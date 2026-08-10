"""
=============================================================================
 GetAround — Module d'analyse des délais (logique métier)
=============================================================================

POURQUOI CE FICHIER EXISTE
--------------------------
On sépare volontairement la LOGIQUE MÉTIER (ce fichier) de l'INTERFACE
(app.py). Avantages, et arguments :
  - testable : on peut valider les calculs sans lancer l'interface ;
  - réutilisable : on peut rejouer ces fonctions dans un notebook ;
  - lisible : l'app ne contient que de l'affichage, pas de calcul.

VOCABULAIRE CLÉ
-------------------------------------------------
  - paire "enchaînée" : une location qui suit une autre location de la
    MÊME voiture, dans un intervalle de moins de 12h.
  - g (gap)   : time_delta_with_previous_rental — l'écart PLANIFIÉ entre la
                fin prévue de la location précédente et le début de celle-ci.
  - d (delay) : delay_at_checkout de la location PRÉCÉDENTE (son retard).
  - impacted  : d > g  -> la voiture précédente est rendue APRÈS l'heure de
                début prévue de la suivante -> le conducteur suivant est gêné.
=============================================================================
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Chemin du fichier de données, construit en relatif par rapport à ce module.
# .parent = dossier "dashboard/". On évite ainsi tout chemin absolu en dur,
# ce qui rend le code portable (machine locale, conteneur Docker, Space HF).
DATA_PATH = Path(__file__).parent / "data" / "get_around_delay_analysis.xlsx"


def load_rentals(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge la feuille brute 'rentals_data' du fichier Excel (21 310 lignes)."""
    return pd.read_excel(path, sheet_name="rentals_data")


def build_chained_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la table des PAIRES ENCHAÎNÉES — c'est le coeur de l'analyse.

    PROBLÈME : chaque ligne connaît l'ID de sa location précédente
    (previous_ended_rental_id), mais le RETARD de cette location précédente
    se trouve dans une AUTRE ligne. Il faut donc aller le chercher.

    SOLUTION : une jointure de la table SUR ELLE-MÊME (self-join).
    On prépare une copie réduite de df où rental_id est renommé en
    previous_ended_rental_id, puis on la recolle sur df. Résultat : chaque
    location courante récupère, dans une nouvelle colonne, le retard de SA
    location précédente.
    """
    # 1) Table "précédente" : on ne garde que l'identifiant + le retard,
    #    et on renomme les colonnes pour qu'elles servent de clé de jointure.
    prev = df[["rental_id", "delay_at_checkout_in_minutes"]].rename(
        columns={
            "rental_id": "previous_ended_rental_id",          # clé de jointure
            "delay_at_checkout_in_minutes": "previous_delay_at_checkout",
        }
    )

    # 2) Jointure INTERNE : ne survivent que les locations qui ONT une
    #    précédente (les autres n'ont pas de previous_ended_rental_id à matcher).
    chain = df.merge(prev, on="previous_ended_rental_id", how="inner")

    # 3) On retire les lignes sans écart planifié connu : sans g, on ne peut
    #    rien simuler. (delay précédent peut rester NaN -> géré plus bas.)
    chain = chain.dropna(subset=["time_delta_with_previous_rental_in_minutes"]).copy()

    # 4) overlap = de combien le retard précédent dépasse l'écart disponible.
    #    overlap > 0  <=>  d > g  <=>  la location courante est IMPACTÉE.
    chain["overlap_minutes"] = (
        chain["previous_delay_at_checkout"]
        - chain["time_delta_with_previous_rental_in_minutes"]
    )
    chain["impacted"] = chain["overlap_minutes"] > 0
    return chain


def headline_metrics(df: pd.DataFrame, chain: pd.DataFrame) -> dict:
    """
    Calcule les indicateurs (KPI) affichés en haut du dashboard.
    Chaque KPI répond à une question que le jury peut poser.
    """
    ended = df[df["state"] == "ended"]                 # locations réellement effectuées
    known = ended["delay_at_checkout_in_minutes"].notna()   # retard connu (sinon NaN)
    # "En retard" = retard strictement positif (négatif = rendu en avance).
    late = ended.loc[known & (ended["delay_at_checkout_in_minutes"] > 0)]

    # Comparaison du taux d'annulation : impactées vs non-impactées.
    # On ne garde que les paires dont le retard précédent est CONNU.
    known_impact = chain.dropna(subset=["previous_delay_at_checkout"])
    cancel_impacted = (
        known_impact.loc[known_impact["impacted"], "state"].eq("canceled").mean()
    )
    cancel_clean = (
        known_impact.loc[~known_impact["impacted"], "state"].eq("canceled").mean()
    )

    return {
        "total_rentals": len(df),                                  # 21 310
        "cancel_rate": (df["state"] == "canceled").mean(),         # ~15 %
        "late_rate": len(late) / known.sum(),                      # ~57 % en retard
        "median_late": late["delay_at_checkout_in_minutes"].median(),  # ~53 min
        "chained_share": len(chain) / len(df),                     # ~9 % enchaînées
        "impacted_rate": known_impact["impacted"].mean(),          # ~12,6 % impactées
        # tuple (taux annul. si impactée, taux annul. sinon) -> 0,17 vs 0,11
        "cancel_lift": (cancel_impacted, cancel_clean),
    }


def simulate(chain: pd.DataFrame, threshold: int, scope: str, total_rentals: int) -> dict:
    """
    Simule l'effet d'un SEUIL (threshold) sur un PÉRIMÈTRE (scope) donné.

    RAISONNEMENT MÉTIER (à savoir réciter) :
      Imposer un délai minimum T rend l'écart effectif = max(g, T).
      Un cas reste un problème si d > max(g, T).
      Donc un cas autrefois problématique (d > g) est RÉSOLU si d <= T.

    Trois quantités en sortie :
      - blocked  (COÛT)    : g < T   -> réservation empêchée par le tampon.
      - impacted (PROBLÈME): d > g   -> cas problématiques actuels.
      - solved   (BÉNÉFICE): impacted ET d <= T -> tampon couvre le retard.
    """
    # Périmètre : toutes les voitures, ou seulement les locations 'connect'.
    sub = chain if scope == "all" else chain[chain["checkin_type"] == "connect"]
    gap = sub["time_delta_with_previous_rental_in_minutes"]   # g
    delay = sub["previous_delay_at_checkout"]                 # d (peut être NaN)

    blocked = int((gap < threshold).sum())            # locations bloquées par le seuil
    impacted_mask = delay > gap                        # NaN > x -> False (cas écarté, OK)
    impacted = int(impacted_mask.sum())
    solved = int((impacted_mask & (delay <= threshold)).sum())

    return {
        "blocked": blocked,
        "blocked_share_total": blocked / total_rentals,            # % de TOUTES les locations
        "blocked_share_scope": blocked / len(sub) if len(sub) else 0.0,
        "impacted": impacted,
        "solved": solved,
        "remaining": impacted - solved,                            # problèmes restants
        "solved_share": solved / impacted if impacted else 0.0,    # % de problèmes résolus
        "scope_size": len(sub),
    }


def tradeoff_curve(chain: pd.DataFrame, scope: str, total_rentals: int,
                   thresholds=range(0, 721, 15)) -> pd.DataFrame:
    """
    Balaie tous les seuils de 0 à 720 min (pas de 15) et renvoie, pour chacun,
    le % de locations bloquées et le % de problèmes résolus.
    C'est ce tableau qui alimente la COURBE D'ARBITRAGE du dashboard.
    """
    rows = []
    for t in thresholds:
        s = simulate(chain, t, scope, total_rentals)
        rows.append(
            {
                "threshold": t,
                "rentals_blocked": s["blocked"],
                "pct_rentals_blocked": s["blocked_share_total"] * 100,
                "problems_solved": s["solved"],
                "pct_problems_solved": s["solved_share"] * 100,
            }
        )
    return pd.DataFrame(rows)
