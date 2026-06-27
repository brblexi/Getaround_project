# Fiche de restitution — Dashboard d'analyse des délais (GetAround, Bloc 5)

> Antisèche pour la soutenance. À lire à voix haute en s'entraînant : si une
> ligne ne se raconte pas naturellement, c'est qu'elle n'est pas encore comprise.

---

## 1. Le problème métier en une phrase

GetAround loue des voitures entre particuliers. Quand un conducteur rend la
voiture **en retard**, ce retard peut **empiéter sur la location suivante** de
la même voiture → friction, voire annulation. Le Product team veut imposer un
**délai minimum entre deux locations**. Mon analyse répond à deux questions :
**quelle valeur de seuil ?** et **sur quel périmètre (toutes les voitures vs
Connect uniquement) ?**

`mobile` = remise des clés en main propre ; `connect` = ouverture par
smartphone, sans contact → « Connect only » est un périmètre plus simple à
imposer techniquement.

---

## 2. Les données (1 fichier, 2 feuilles)

`rentals_data` — 21 310 locations. Colonnes utiles :

| Colonne | Sens |
|---|---|
| `state` | `ended` (effectuée) / `canceled` |
| `checkin_type` | `mobile` / `connect` |
| `delay_at_checkout_in_minutes` | retard au retour (négatif = rendu en avance) |
| `previous_ended_rental_id` | ID de la location précédente de la même voiture (si < 12h) |
| `time_delta_with_previous_rental_in_minutes` | écart **planifié** avec la précédente |

**Choix de nettoyage à défendre :**
- Outliers de retard extrêmes (−22 433 à +71 084 min) → **clip à ±300 min
  uniquement pour l'affichage** de l'histogramme. La logique de simulation
  garde les valeurs brutes (un retard de 5 h reste un retard de 5 h).
- `NaN` de retard = locations annulées (pas de checkout) → exclues des stats de
  retard, mais comptées dans le total.

---

## 3. Le cœur de l'analyse : la jointure auto (LE point à maîtriser)

Chaque ligne connaît l'**ID** de sa location précédente, mais pas son
**retard** (qui est dans une autre ligne). Je fais donc une **jointure de la
table sur elle-même** pour rapatrier le retard précédent à côté de la location
courante.

Ensuite, pour chaque paire enchaînée :
- `g` = écart planifié disponible
- `d` = retard de la location précédente
- **`impacted` ⇔ `d > g`** : la voiture précédente est rendue après l'heure de
  début prévue de la suivante → conducteur suivant gêné.

→ **218 cas impactés sur 1 729 paires exploitables (12,6 %).**

---

## 4. Les 3 métriques de la simulation

Pour un seuil `T` et un périmètre :

| Métrique | Définition | Rôle |
|---|---|---|
| `blocked` | `g < T` | **COÛT** — réservation empêchée par le tampon |
| `impacted` | `d > g` | **PROBLÈME** actuel |
| `solved` | `impacted` ET `d ≤ T` | **BÉNÉFICE** — tampon couvre le retard |

**La phrase à réciter :** imposer un délai minimum `T` rend l'écart effectif
égal à `max(g, T)`. Un cas reste un problème si `d > max(g, T)`. Donc un cas
autrefois problématique (`d > g`) est résolu dès que `d ≤ T`.

---

## 5. Résultats chiffrés (toutes voitures)

| Seuil | Locations bloquées | % de toutes les locations | % problèmes résolus |
|---|---|---|---|
| 60 min | 401 | 1,9 % | 47 % |
| 120 min | 666 | 3,1 % | 67 % |
| 180 min | 870 | 4,1 % | 77 % |

- Courbe **verte** (résolus) : monte vite puis **plafonne**.
- Courbe **rouge** (bloquées) : monte lentement, presque linéaire.
- → **rendements décroissants au-delà de ~2 h.**
- **Connect only** bloque ~2× moins, mais laisse les cas `mobile`
  (majoritaires) non traités.

**Recommandation :** **60–120 min sur toutes les voitures** (≈ 47–67 % résolus
pour < 3 % bloquées), à **valider par A/B test**.

---

## 6. Industrialisation (objet du Bloc 5)

| Choix | Justification |
|---|---|
| **Streamlit** | data app en Python pur, pas de front JS, rapide à livrer |
| `app.py` / `utils.py` séparés | logique métier isolée → testable + réutilisable |
| `@st.cache_data` | charge/calcule une seule fois, pas à chaque slider |
| **Docker** | image reproductible, déployable partout |
| **HF Spaces** (`sdk: docker`, port 7860) | build auto depuis le Dockerfile, gratuit |
| Tests **AppTest** | validation automatisée : l'app se rend sans exception |

---

## 7. Limites à assumer spontanément

- Échantillon de paires exploitables petit (1 729).
- **Corrélation** impact ↔ annulation (17 % vs 11 %), pas causalité prouvée.
- Une location « bloquée » n'est pas forcément un revenu perdu (report possible).
- Pas d'horodatage absolu → raisonnement en deltas relatifs.

---

## 8. Questions probables du jury → réponse courte

- *« Pourquoi seulement 9 % de l'effet ? »* → seules ~9 % des locations sont
  enchaînées dos-à-dos ; le reste n'est pas concerné par la feature.
- *« Pourquoi pas Connect only puisque ça bloque moins ? »* → parce que les
  retards se concentrent sur le `mobile` ; Connect only laisserait la majorité
  des problèmes non résolus.
- *« Comment as-tu obtenu le retard de la location précédente ? »* → jointure de
  la table sur elle-même via `previous_ended_rental_id`.
- *« Pourquoi clipper à ±300 min ? »* → uniquement pour la lisibilité de
  l'histogramme ; les calculs utilisent les valeurs brutes.
