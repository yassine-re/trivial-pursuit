#!/usr/bin/env python3
"""Nettoie la couche Bronze OpenTDB et produit le Parquet Silver.

La transformation est déterministe : une question conserve le même identifiant
et le même ordre de choix même si l'ordre des lignes du CSV source change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "bronze" / "questions_raw.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "silver" / "questions_clean.parquet"

RAW_COLUMNS = (
    "category",
    "type",
    "difficulty",
    "question",
    "correct_answer",
    "incorrect_answers",
)
ALLOWED_TYPES = {"boolean": 1, "multiple": 3}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}

SILVER_SCHEMA = pa.schema(
    [
        pa.field("source_row_number", pa.int32(), nullable=False),
        pa.field("question_id", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("type", pa.string(), nullable=False),
        pa.field("difficulty", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("correct_answer", pa.string(), nullable=False),
        pa.field("incorrect_answers", pa.list_(pa.string()), nullable=False),
        pa.field("answer_choices", pa.list_(pa.string()), nullable=False),
        pa.field("correct_choice", pa.string(), nullable=False),
    ]
)


class DataQualityError(ValueError):
    """Signale une ligne Bronze qui ne respecte pas le contrat OpenTDB."""


@dataclass
class TransformationStats:
    source_rows: int = 0
    output_rows: int = 0
    cleaned_values: int = 0
    exact_duplicates_removed: int = 0


def clean_text(value: Any, field: str, row_number: int) -> str:
    """Décode les entités HTML et retire les espaces en bordure."""

    if not isinstance(value, str):
        raise DataQualityError(
            f"Ligne {row_number} : {field} doit être une chaîne de caractères."
        )
    cleaned = html.unescape(value).strip()
    if not cleaned:
        raise DataQualityError(f"Ligne {row_number} : {field} est vide.")
    return cleaned


def parse_incorrect_answers(raw_value: str, row_number: int) -> list[str]:
    """Valide et nettoie la liste JSON des mauvaises réponses."""

    try:
        values = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as error:
        raise DataQualityError(
            f"Ligne {row_number} : incorrect_answers n'est pas un JSON valide."
        ) from error

    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise DataQualityError(
            f"Ligne {row_number} : incorrect_answers doit être une liste de chaînes."
        )
    return [clean_text(value, "incorrect_answers", row_number) for value in values]


def build_question_id(record: dict[str, Any]) -> str:
    """Construit une clé sémantique stable, indépendante de l'ordre du CSV."""

    identity = {
        "category": record["category"],
        "type": record["type"],
        "difficulty": record["difficulty"],
        "question": record["question"],
        "correct_answer": record["correct_answer"],
        # L'ordre fourni par l'API n'a pas de signification métier.
        "incorrect_answers": sorted(record["incorrect_answers"]),
    }
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_answer_choices(
    question_id: str,
    correct_answer: str,
    incorrect_answers: list[str],
) -> tuple[list[str], str]:
    """Mélange les réponses de façon reproductible et retourne la bonne lettre."""

    choices = [correct_answer, *incorrect_answers]
    random.Random(int(question_id, 16)).shuffle(choices)
    correct_index = choices.index(correct_answer)
    return choices, string.ascii_uppercase[correct_index]


def transform_row(
    raw_row: dict[str, str],
    row_number: int,
) -> tuple[dict[str, Any], int]:
    """Transforme et valide une ligne issue du CSV Bronze."""

    if None in raw_row:
        raise DataQualityError(
            f"Ligne {row_number} : le nombre de colonnes dépasse l'en-tête CSV."
        )

    cleaned = {
        field: clean_text(raw_row[field], field, row_number)
        for field in RAW_COLUMNS[:-1]
    }
    incorrect_answers = parse_incorrect_answers(
        raw_row["incorrect_answers"], row_number
    )
    cleaned["incorrect_answers"] = incorrect_answers

    question_type = cleaned["type"]
    if question_type not in ALLOWED_TYPES:
        raise DataQualityError(
            f"Ligne {row_number} : type inconnu {question_type!r}."
        )
    expected_answers = ALLOWED_TYPES[question_type]
    if len(incorrect_answers) != expected_answers:
        raise DataQualityError(
            f"Ligne {row_number} : {question_type} exige {expected_answers} "
            f"mauvaise(s) réponse(s), {len(incorrect_answers)} reçue(s)."
        )
    if cleaned["difficulty"] not in ALLOWED_DIFFICULTIES:
        raise DataQualityError(
            f"Ligne {row_number} : difficulté inconnue "
            f"{cleaned['difficulty']!r}."
        )

    all_answers = [cleaned["correct_answer"], *incorrect_answers]
    normalized_answers = {answer.casefold() for answer in all_answers}
    if len(normalized_answers) != len(all_answers):
        raise DataQualityError(
            f"Ligne {row_number} : les choix de réponse ne sont pas uniques."
        )

    question_id = build_question_id(cleaned)
    answer_choices, correct_choice = make_answer_choices(
        question_id,
        cleaned["correct_answer"],
        incorrect_answers,
    )

    cleaned_values = sum(
        raw_row[field] != cleaned[field] for field in RAW_COLUMNS[:-1]
    )
    original_incorrect = json.loads(raw_row["incorrect_answers"])
    cleaned_values += sum(
        original != new
        for original, new in zip(original_incorrect, incorrect_answers, strict=True)
    )

    return (
        {
            "source_row_number": row_number,
            "question_id": question_id,
            **cleaned,
            "answer_choices": answer_choices,
            "correct_choice": correct_choice,
        },
        cleaned_values,
    )


def transform_csv(input_path: Path) -> tuple[list[dict[str, Any]], TransformationStats]:
    """Charge le Bronze, contrôle son schéma et retourne les lignes Silver."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Fichier Bronze introuvable : {input_path}")

    rows: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    stats = TransformationStats()

    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise DataQualityError(
                "En-tête Bronze invalide. "
                f"Attendu : {list(RAW_COLUMNS)} ; reçu : {reader.fieldnames}."
            )

        for row_number, raw_row in enumerate(reader, start=2):
            stats.source_rows += 1
            row, cleaned_count = transform_row(raw_row, row_number)
            stats.cleaned_values += cleaned_count
            if row["question_id"] in seen_question_ids:
                stats.exact_duplicates_removed += 1
                continue
            seen_question_ids.add(row["question_id"])
            rows.append(row)

    if not rows:
        raise DataQualityError("Le fichier Bronze ne contient aucune question.")
    stats.output_rows = len(rows)
    return rows, stats


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
    input_path: Path,
) -> None:
    """Écrit le Silver de manière atomique avec son lignage en métadonnées."""

    metadata = {
        b"medallion_layer": b"silver",
        b"dataset_kind": b"clean_questions",
        b"silver_schema_version": b"1",
        b"source_file": display_source_path(input_path).encode("utf-8"),
        b"source_sha256": file_sha256(input_path).encode("ascii"),
        b"transformation": b"bronze_to_silver_v1",
    }
    table = pa.Table.from_pylist(
        rows,
        schema=SILVER_SCHEMA.with_metadata(metadata),
    )
    null_count = sum(column.null_count for column in table.columns)
    if table.num_rows != len(rows) or null_count != 0:
        raise DataQualityError("Le contrôle final de la table Silver a échoué.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
        # Relire les métadonnées avant de remplacer une éventuelle sortie valide.
        parquet_file = pq.ParquetFile(temporary_path)
        if parquet_file.metadata.num_rows != len(rows):
            raise DataQualityError("Le Parquet temporaire est incomplet.")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_silver(input_path: Path, output_path: Path) -> TransformationStats:
    rows, stats = transform_csv(input_path)
    write_parquet(rows, output_path, input_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_silver(args.input, args.output)
    output_size = args.output.stat().st_size
    print(f"Source Bronze : {args.input}")
    print(f"Lignes lues : {stats.source_rows}")
    print(f"Valeurs nettoyées : {stats.cleaned_values}")
    print(f"Doublons stricts retirés : {stats.exact_duplicates_removed}")
    print(f"Lignes Silver : {stats.output_rows}")
    print(f"Silver créé : {args.output} ({output_size:,} octets)")


if __name__ == "__main__":
    main()
