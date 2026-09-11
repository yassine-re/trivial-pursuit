# Trivial Pursuit — benchmark de modèles IA

Pipeline de data engineering pour mesurer les performances de modèles locaux sur
les questions de culture générale d'Open Trivia Database.

## Architecture actuelle

```text
data/
├── bronze/
│   └── questions_raw.csv
├── silver/
│   ├── questions_clean.parquet
│   └── benchmark_results.parquet       # créé après un benchmark LM Studio
└── gold/
    └── benchmark.duckdb                # construit par dbt
src/
├── extract/extract_opentdb.py
├── transform/build_silver.py
├── enrich/
│   ├── prompts.py
│   └── run_benchmark.py
└── dashboard/                          # lecture et graphiques
dbt/models/marts/fct_benchmark_results.sql
streamlit_app.py                        # interface Streamlit
tests/
```

- **Bronze** conserve exactement les données renvoyées par OpenTDB.
- **Silver questions** nettoie et type les questions dans un Parquet Zstandard.
- **Silver benchmark** conserve une ligne par question, modèle, prompt et
  configuration de génération, y compris la réponse brute et le temps mesuré.
- **Gold** est construit avec dbt dans DuckDB sous forme de marts analytiques.
- **Dashboard** explore les résultats Gold avec Streamlit et Plotly.

## Installation

Python 3.10 ou plus récent est requis.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Construire la couche Silver propre

```bash
make silver
```

Équivalent sans Make :

```bash
python -m src.transform.build_silver
```

La transformation :

- valide le contrat du CSV et les cardinalités des QCM/Vrai-Faux ;
- décode les entités HTML et retire les espaces en bordure ;
- retire uniquement les doublons sémantiquement stricts ;
- crée un `question_id` SHA-256 stable ;
- mélange les choix de manière déterministe ;
- inscrit le chemin et l'empreinte SHA-256 du Bronze dans les métadonnées ;
- écrit atomiquement `data/silver/questions_clean.parquet`.

## Enrichir avec LM Studio

1. Installer [LM Studio](https://lmstudio.ai/) et télécharger un modèle.
2. Dans l'onglet **Developer**, démarrer le serveur local. Il peut aussi être
   lancé avec la CLI :

```bash
lms server start
```

3. Activer le chargement JIT ou charger le modèle manuellement. L'identifiant
   visible dans LM Studio doit correspondre à celui passé au benchmark.

Le pipeline utilise l'API OpenAI-compatible de LM Studio, disponible par défaut
sur `http://localhost:1234/v1`. Le modèle par défaut est
`google/gemma-3-4b`. Vérifier les modèles visibles par le serveur avec :

```bash
curl http://localhost:1234/v1/models
```

Commencer par un petit essai :

```bash
make benchmark LIMIT=20
```

Pour utiliser un autre modèle, reprendre exactement son identifiant LM Studio :

```bash
make benchmark MODEL="identifiant-du-modèle" LIMIT=20
```

Puis lancer le dataset complet avec le modèle par défaut ou un modèle choisi :

```bash
make benchmark
make benchmark MODEL="identifiant-du-modèle"
```

La commande reprend automatiquement un fichier existant, rejoue les erreurs et
enregistre un checkpoint toutes les dix questions. Une interruption clavier
conserve également la progression. Pour rejouer les résultats déjà réussis :

```bash
python -m src.enrich.run_benchmark --model "identifiant-du-modèle" --force
```

Les variables `LM_STUDIO_URL` et `LM_STUDIO_MODEL` permettent de modifier les
valeurs par défaut. Si l'authentification du serveur local est activée, définir
également `LM_STUDIO_API_TOKEN`.

Le prompt est versionné par `prompt_id`. La première version exige uniquement la
lettre du choix ; elle ne transmet jamais la bonne réponse au modèle.

## Contrôles

```bash
make test
```

Les tests vérifient le nettoyage, la stabilité des identifiants et des choix, les
erreurs de qualité, les métadonnées Parquet, le parsing des réponses IA et le
schéma du résultat de benchmark.

## Construire la couche Gold avec dbt

```bash
make dbt-debug
make dbt-build
```

Ces commandes calculent un chemin absolu vers la racine du projet et le passent
à dbt avec `BI_BENCHMARK_ROOT`. Les emplacements de la base DuckDB et du
Parquet Silver ne dépendent donc pas du répertoire courant. La variable peut
également être fournie explicitement pour lancer dbt sans Make.

Chaque mart conserve le grain complet d'une configuration de benchmark :
`model_name`, `prompt_id`, `generation_temperature` et `generation_seed`.
Les indicateurs distinguent :

- `overall_accuracy` : bonnes réponses divisées par toutes les tentatives ;
- `valid_answer_accuracy` : bonnes réponses divisées par les réponses valides ;
- `valid_answer_rate`, `invalid_answer_rate` et `technical_error_rate` : qualité
  d'exécution du benchmark.

## Dashboard interactif Streamlit

Installer les dépendances dans l'environnement virtuel, actualiser la Gold,
puis démarrer l'interface :

```bash
.venv/bin/python -m pip install -r requirements.txt
make dbt-build
make dashboard
```

Ouvrir [http://localhost:8501](http://localhost:8501). Le serveur écoute uniquement
en local. `Ctrl+C` dans le terminal permet de l'arrêter. `make dashboard` fonctionne
sans activer le venv ; depuis un autre répertoire, utiliser
`make -C /chemin/vers/bi-ai-benchmark dashboard`.

Le dashboard contient quatre vues :

- **Comparatif** : classement, précision / latence, face-à-face question par
  question et export CSV des indicateurs.
- **Par thème** : heatmap des catégories et graphiques par difficulté ou type,
  avec les effectifs au survol.
- **Exécution** : réponses correctes, incorrectes, invalides et erreurs ; latence
  moyenne, médiane, P95 et consommation de tokens.
- **Explorer les réponses** : recherche textuelle, filtre par résultat, tableau
  sélectionnable, choix proposés, réponse brute et export CSV du périmètre.

Choisir une configuration réelle (prompt, température et seed), puis les modèles.
Le mode **Questions communes uniquement**, activé par défaut, conserve les
questions tentées par tous les modèles sélectionnés, y compris leurs erreurs.
Les filtres catégorie, difficulté et type vides signifient « tout inclure » ;
une sélection de modèles vide affiche un message.

Les indicateurs sont recalculés depuis les faits filtrés, jamais par une moyenne
de pourcentages. L'accuracy sur réponses valides est indéfinie sans réponse valide.
Les moyennes de temps incluent les relances et erreurs ; les moyennes de tokens
ignorent les valeurs absentes. Un résultat `success` signifie une réponse
interprétable, pas nécessairement correcte.

Le dashboard lit **`main.fct_benchmark_results`**, une table détaillée matérialisée
par dbt, avec une connexion DuckDB courte en lecture seule. Les graphiques et les
exports utilisent le même instantané. Après chaque nouveau benchmark :

```bash
make dbt-build
```

Puis cliquer sur **Actualiser les données**. Une alerte signale un Parquet Silver
plus récent que l'instantané Gold. Le dashboard ne lance pas d'inférence et ne
reconstruit pas dbt lors des interactions.

Si DuckDB signale un verrou, fermer sa session CLI avec `.exit` (ou attendre la
fin du build). La table manquante et le schéma ancien sont signalés avec la
commande à exécuter. `BI_BENCHMARK_DB=/chemin/absolu/base.duckdb` permet de choisir
une autre base, pour dbt comme pour Streamlit ; `BI_BENCHMARK_ROOT` conserve son
rôle pour localiser les données Silver.

Les tests du dashboard se lancent avec :

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Références : [cache Streamlit](https://docs.streamlit.io/develop/concepts/architecture/caching),
[tests d'application](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
