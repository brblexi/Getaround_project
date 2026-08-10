---
title: GetAround Delay Analysis
emoji: 🚗
colorFrom: purple
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# GetAround — Dashboard d'analyse des délais

Dashboard interactif qui quantifie l'impact des retours tardifs sur la location
suivante, et simule l'effet d'un **délai minimum entre deux locations** ainsi que
son périmètre d'application (toutes les voitures ou Connect uniquement).

Construit avec Streamlit et Plotly, servi via Docker sur Hugging Face Spaces.

**URL de production :** https://alexbarbier-getaround-dashboard.hf.space

## Ce que montre le dashboard

1. **Fréquence des retards** — distribution et gravité des délais au retour.
2. **Impact sur le conducteur suivant** — taux d'impact par type de check-in et
   surcroît d'annulations.
3. **Simulateur de seuil** — arbitrage en direct entre problèmes résolus et
   locations bloquées.
4. **Lecture des données** — plage de seuil recommandée pour un test A/B.

## Contenu du dossier

```
dashboard/
├── app.py           # interface Streamlit (affichage uniquement)
├── utils.py         # chargement des données et calculs métier
├── data/            # get_around_delay_analysis.xlsx
├── .streamlit/      # configuration (thème, options serveur)
├── requirements.txt # dépendances Python
├── Dockerfile       # environnement d'exécution
├── FICHE_JURY.md    # note de synthèse
└── README.md        # ce fichier (son entête YAML configure le Space)
```

La séparation `app.py` / `utils.py` est délibérée : le premier ne fait que de
l'affichage, le second porte toute la logique de calcul. Cela rend les métriques
testables indépendamment de l'interface.

---

# Utilisation

Trois façons de faire tourner ce dashboard, de la plus légère à la plus proche de
la production. Les étapes 1 et 2 servent à valider avant de déployer à l'étape 3.

## 1. En local, sans Docker

```bash
cd dashboard/
pip install -r requirements.txt
streamlit run app.py
```

Streamlit ouvre automatiquement le navigateur sur **http://localhost:8501**, son
port par défaut.

## 2. En local, avec Docker

Reproduit exactement l'environnement du Space. Si cette étape passe, le
déploiement passera.

```bash
cd dashboard/
docker build -t getaround-dashboard .
docker run -p 7860:7860 --rm getaround-dashboard
```

Ouvrir ensuite **http://localhost:7860**.

> **Pourquoi 7860 ici et 8501 juste au-dessus ?** Hugging Face Spaces attend que
> l'application écoute sur le port déclaré dans `app_port` de l'entête YAML. Le
> `Dockerfile` fige donc ce port dans sa commande de démarrage :
>
> ```dockerfile
> CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
> ```
>
> Le `--server.address 0.0.0.0` est indispensable : par défaut Streamlit n'accepte
> que les connexions venant de l'intérieur du conteneur, et le proxy de Hugging
> Face ne pourrait pas le joindre.
>
> Le `-p 7860:7860` mappe ce port du conteneur vers le même port de ta machine.
> Attention : ce port est aussi celui du conteneur de l'API. Ne pas lancer les
> deux simultanément avec le même mappage, sinon `localhost:7860` renvoie
> l'application démarrée en premier. Utiliser `-p 7861:7860` pour l'une des deux.

## 3. Déployer sur Hugging Face Spaces

1. **Créer un Space → SDK : `Docker`.**

   Le « SDK » indique à HF comment exécuter l'application. Le mode `Streamlit`
   existe et suffirait pour un cas standard, mais `Docker` est retenu ici pour
   maîtriser la version de Python et les dépendances exactement comme en local.

2. **Pousser le *contenu* du dossier `dashboard/`** à la racine du Space — pas le
   dossier lui-même. Ce README et son entête YAML doivent se retrouver à la
   racine, et le fichier de données doit être présent dans `data/`.

   ```bash
   git clone https://huggingface.co/spaces/AlexBarbier/getaround-dashboard
   cd getaround-dashboard

   git lfs install
   git lfs track "*.xlsx"            # avant d'ajouter le fichier de données

   cp -r ../Projet_GetAround/dashboard/. .

   git add -A
   git commit -m "Deploy GetAround delay analysis dashboard"
   git push
   ```

   Sous PowerShell, la copie s'écrit :
   `Copy-Item -Path ..\Projet_GetAround\dashboard\* -Destination . -Recurse -Force`

3. **Le Space construit l'image et démarre le conteneur.** Le statut passe en
   *Building* ; l'onglet **Logs** affiche la sortie du build puis celle de
   Streamlit.

> Les logs du Space affichent une ligne `Local URL: http://localhost:7860`. Cette
> adresse est celle de l'intérieur du conteneur, sur les serveurs de Hugging Face
> — elle n'est pas accessible depuis ton navigateur. L'adresse publique est celle
> indiquée en haut de ce README.

---

# Lecture des données

**Le levier est réel mais étroit.** Environ 9 % des locations s'enchaînent sur la
même voiture dans les 12 heures : c'est la seule population que la fonctionnalité
touche. Mais à l'intérieur de ce sous-ensemble, les retours tardifs font
nettement monter les annulations — d'environ 11 % à 17 %.

**Les retours diminuent vite.** Augmenter le seuil continue de résoudre des cas,
mais chaque tranche de minutes supplémentaire en résout de moins en moins tout en
continuant de bloquer des réservations. La zone de rendement décroissant commence
vers deux heures.

**Le périmètre compte plus que la largeur.** Les retards se concentrent sur les
check-in *mobile*. Une règle limitée à *Connect* bloque donc très peu de
réservations, mais laisse intacte la majorité des cas problématiques. Un seuil de
**60 à 120 minutes sur toutes les voitures** couvre environ la moitié à deux tiers
des problèmes en bloquant moins de 3 % des locations — point de départ défendable
pour un test A/B.

# Limites connues

**Analyse observationnelle.** Les chiffres décrivent ce qui s'est produit, pas ce
qui se produirait après mise en place du seuil. Une réservation « bloquée » n'est
pas nécessairement une location perdue : le conducteur peut décaler son créneau
ou choisir une autre voiture. Le nombre de locations bloquées est donc une borne
haute du coût réel.

**Fenêtre d'enchaînement fixée à 12 heures.** Le seuil retenu pour considérer que
deux locations se suivent est un choix de modélisation, pas une donnée. Le faire
varier modifie la taille de la population concernée.

**Valeurs extrêmes.** L'histogramme des délais est tronqué à ±5 h pour rester
lisible, mais les simulations utilisent les valeurs brutes, y compris des retards
de plusieurs dizaines de milliers de minutes qui relèvent probablement d'erreurs
de saisie.

**Données figées.** Le fichier Excel est embarqué dans l'image Docker. Toute mise
à jour des données impose de reconstruire et redéployer le Space.

# Évolutions possibles

- **Tester la sensibilité** des conclusions à la fenêtre d'enchaînement et au
  traitement des valeurs extrêmes.
- **Distinguer les annulations** dues à un retard de celles ayant une autre cause,
  pour resserrer l'estimation du bénéfice.
- **Segmenter par ville ou par type de véhicule**, la concentration des
  enchaînements pouvant varier fortement selon le marché local.
- **Découpler les données de l'image** en les chargeant depuis une source externe,
  pour actualiser l'analyse sans redéployer.

---

Source : `get_around_delay_analysis.xlsx`. Une location est dite *impactée* quand
la voiture précédente est rendue après son heure de départ prévue ; un problème
est *résolu* quand le délai imposé couvre le retard du conducteur précédent.
