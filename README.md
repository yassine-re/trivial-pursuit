# Trivial Pursuit — benchmark de modèles IA

Pipeline de data engineering pour mesurer les performances de modèles locaux sur
les questions de culture générale d'Open Trivia Database.

## Architecture actuelle

```text
data/
├── bronze/
│   └── questions_raw.csv
└── silver/
    ├── questions_clean.parquet
    └── benchmark_results.parquet       # créé après un benchmark Ollama
src/
├── extract/extract_opentdb.py
├── transform/build_silver.py
└── enrich/
    ├── prompts.py
    └── run_benchmark.py
tests/
```

- **Bronze** conserve exactement les données renvoyées par OpenTDB.
- **Silver questions** nettoie et type les questions dans un Parquet Zstandard.
- **Silver benchmark** conserve une ligne par question, modèle, prompt et
  configuration de génération, y compris la réponse brute et le temps mesuré.
- **Gold** sera construit ensuite avec dbt dans DuckDB.

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

## Enrichir avec Ollama

Une fois Ollama démarré et un modèle téléchargé, commencer par un petit essai :

```bash
make benchmark MODEL=llama3.2:3b LIMIT=20
```

Puis lancer le dataset complet :

```bash
make benchmark MODEL=llama3.2:3b
```

La commande reprend automatiquement un fichier existant, rejoue les erreurs et
enregistre un checkpoint toutes les dix questions. Une interruption clavier
conserve également la progression. Pour rejouer les résultats déjà réussis :

```bash
python -m src.enrich.run_benchmark --model llama3.2:3b --force
```

Le prompt est versionné par `prompt_id`. La première version exige uniquement la
lettre du choix ; elle ne transmet jamais la bonne réponse au modèle.

## Contrôles

```bash
make test
```

Les tests vérifient le nettoyage, la stabilité des identifiants et des choix, les
erreurs de qualité, les métadonnées Parquet, le parsing des réponses IA et le
schéma du résultat de benchmark.
