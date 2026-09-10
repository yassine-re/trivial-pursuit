import csv
import json
import time
from collections import Counter
from pathlib import Path

import requests


API_URL = "https://opentdb.com/api.php"
TOKEN_URL = "https://opentdb.com/api_token.php"
COUNT_URL = "https://opentdb.com/api_count_global.php"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "bronze" / "questions_raw.csv"
CHECKPOINT_FILE = (
    PROJECT_ROOT / "data" / "bronze" / "questions_raw.checkpoint.json"
)

BATCH_SIZE = 50
RATE_LIMIT_SECONDS = 5.1
MAX_REQUEST_ATTEMPTS = 5


def get_json(url, params=None):
    """Appelle l'API avec des reprises sur les erreurs réseau temporaires."""

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            status_code = error.response.status_code
            server = error.response.headers.get("Server", "")

            if status_code == 403 and server.lower() == "cato":
                raise RuntimeError(
                    "Accès à OpenTDB bloqué par la politique réseau Cato."
                ) from error

            if status_code < 500 and status_code != 429:
                raise

            if attempt == MAX_REQUEST_ATTEMPTS:
                raise

            print(
                f"Erreur HTTP temporaire ({status_code}), nouvelle tentative "
                f"{attempt + 1}/{MAX_REQUEST_ATTEMPTS}..."
            )
            time.sleep(RATE_LIMIT_SECONDS)
        except requests.RequestException as error:
            if attempt == MAX_REQUEST_ATTEMPTS:
                raise

            print(
                f"Erreur réseau temporaire ({error.__class__.__name__}), "
                f"nouvelle tentative {attempt + 1}/{MAX_REQUEST_ATTEMPTS}..."
            )
            time.sleep(RATE_LIMIT_SECONDS)

    raise RuntimeError("Nombre maximal de tentatives atteint.")


def get_total_questions():
    """Récupère le nombre de questions vérifiées disponibles sur OpenTDB."""

    data = get_json(COUNT_URL)

    return data["overall"]["total_num_of_verified_questions"]


def get_session_token():
    """Crée un token OpenTDB permettant d'éviter les doublons."""

    data = get_json(
        TOKEN_URL,
        params={"command": "request"},
    )

    if data["response_code"] != 0:
        raise RuntimeError("Impossible de récupérer un token OpenTDB.")

    return data["token"]


def fetch_questions(token, total_questions, questions=None):
    """Récupère toutes les questions OpenTDB par lots."""

    questions = list(questions or [])

    while len(questions) < total_questions:

        remaining = total_questions - len(questions)
        amount = min(BATCH_SIZE, remaining)

        params = {
            "amount": amount,
            "token": token,
        }

        data = get_json(
            API_URL,
            params=params,
        )
        response_code = data["response_code"]

        if response_code == 0:
            batch = data["results"]
            questions.extend(batch)
            save_checkpoint(token, total_questions, questions)

            print(
                f"{len(questions)}/{total_questions} questions récupérées"
            )

        elif response_code == 5:
            print("Rate limit atteint, nouvelle tentative...")
            time.sleep(RATE_LIMIT_SECONDS)
            continue

        elif response_code == 4:
            print(
                "Toutes les questions disponibles pour ce token ont été "
                "récupérées."
            )
            break

        else:
            raise RuntimeError(
                f"Erreur OpenTDB : response_code={response_code}"
            )

        if len(questions) < total_questions:
            time.sleep(RATE_LIMIT_SECONDS)

    return questions


def load_checkpoint(total_questions):
    """Recharge une extraction interrompue tant que son token reste valide."""

    if not CHECKPOINT_FILE.exists():
        return None

    try:
        checkpoint = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("Checkpoint illisible : une nouvelle extraction sera effectuée.")
        return None

    if (
        checkpoint.get("total_questions") != total_questions
        or not isinstance(checkpoint.get("token"), str)
        or not isinstance(checkpoint.get("questions"), list)
        or time.time() - checkpoint.get("updated_at", 0) >= 6 * 60 * 60
    ):
        return None

    questions = checkpoint["questions"]
    print(
        f"Reprise du checkpoint : {len(questions)}/{total_questions} questions."
    )
    return checkpoint["token"], questions


def save_checkpoint(token, total_questions, questions):
    """Enregistre atomiquement la progression après chaque lot."""

    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = CHECKPOINT_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(
            {
                "token": token,
                "total_questions": total_questions,
                "updated_at": time.time(),
                "questions": questions,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_file.replace(CHECKPOINT_FILE)


def save_to_csv(questions):
    """Sauvegarde les données brutes dans la couche Bronze."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "category",
            "type",
            "difficulty",
            "question",
            "correct_answer",
            "incorrect_answers",
        ])

        for question in questions:
            writer.writerow([
                question["category"],
                question["type"],
                question["difficulty"],
                question["question"],
                question["correct_answer"],
                json.dumps(
                    question["incorrect_answers"],
                    ensure_ascii=False,
                ),
            ])

    temporary_file.replace(OUTPUT_FILE)


def main():

    print("Récupération du nombre de questions OpenTDB...")

    total_questions = get_total_questions()

    print(f"{total_questions} questions vérifiées disponibles.")

    checkpoint = load_checkpoint(total_questions)

    if checkpoint is None:
        # On respecte le rate limit entre les appels
        time.sleep(RATE_LIMIT_SECONDS)

        print("Création du session token...")

        token = get_session_token()
        questions = []

        print("Token récupéré.")

        time.sleep(RATE_LIMIT_SECONDS)
    else:
        token, questions = checkpoint

    print("Début de l'extraction...")

    questions = fetch_questions(
        token,
        total_questions,
        questions,
    )

    print("\nVérification des données...")

    multiple_count = sum(
        question["type"] == "multiple"
        for question in questions
    )

    boolean_count = sum(
        question["type"] == "boolean"
        for question in questions
    )

    question_counts = Counter(
        question["question"]
        for question in questions
    )
    repeated_question_count = sum(
        count - 1
        for count in question_counts.values()
        if count > 1
    )

    # L'API ne fournit pas d'identifiant de question. Deux entrées peuvent avoir
    # le même texte tout en différant par leur catégorie, difficulté ou réponses.
    # On compte donc séparément les lignes entièrement identiques.
    row_counts = Counter(
        json.dumps(question, sort_keys=True, ensure_ascii=False)
        for question in questions
    )
    exact_duplicate_count = sum(
        count - 1
        for count in row_counts.values()
        if count > 1
    )

    print(f"Total récupéré : {len(questions)}")
    print(f"QCM : {multiple_count}")
    print(f"Vrai/Faux : {boolean_count}")
    print(f"Textes de question répétés : {repeated_question_count}")
    print(f"Lignes strictement identiques : {exact_duplicate_count}")

    if len(questions) != total_questions:
        raise RuntimeError(
            f"Extraction incomplète : "
            f"{len(questions)}/{total_questions} questions."
        )

    if repeated_question_count:
        print(
            "Avertissement : les répétitions sont conservées dans la couche "
            "Bronze afin de préserver les données sources."
        )

    save_to_csv(questions)
    CHECKPOINT_FILE.unlink(missing_ok=True)

    print(f"\n✅ Couche Bronze créée : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
