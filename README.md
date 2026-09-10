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
    └── benchmark_results.parquet       # créé après un benchmark LM Studio
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
