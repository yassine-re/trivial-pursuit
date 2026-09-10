#!/usr/bin/env python3
"""Interroge Ollama et enrichit la couche Silver avec ses réponses brutes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.enrich.prompts import PROMPT_TEMPLATES, render_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "silver" / "questions_clean.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "silver" / "benchmark_results.parquet"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

QUESTION_COLUMNS = (
    "source_row_number",
    "question_id",
    "category",
    "type",
    "difficulty",
    "question",
    "correct_answer",
    "incorrect_answers",
    "answer_choices",
    "correct_choice",
)

RESULT_SCHEMA = pa.schema(
    [
        pa.field("benchmark_id", pa.string(), nullable=False),
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
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("prompt_id", pa.string(), nullable=False),
        pa.field("prompt", pa.string(), nullable=False),
        pa.field("generation_temperature", pa.float32(), nullable=False),
        pa.field("generation_seed", pa.int32(), nullable=False),
        pa.field("ai_answer_raw", pa.string()),
        pa.field("ai_answer", pa.string()),
        pa.field("ai_correct", pa.bool_(), nullable=False),
        pa.field("response_time", pa.float64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error_message", pa.string()),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("ollama_total_duration_seconds", pa.float64()),
        pa.field("ollama_load_duration_seconds", pa.float64()),
        pa.field("prompt_eval_count", pa.int32()),
        pa.field("eval_count", pa.int32()),
    ]
)


class OllamaError(RuntimeError):
    """Erreur explicite de communication ou de réponse Ollama."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        max_attempts: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        normalized_url = base_url.strip()
        if "://" not in normalized_url:
            normalized_url = f"http://{normalized_url}"
        self.base_url = normalized_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.session = session or requests.Session()

    def installed_models(self) -> list[str]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags", timeout=min(self.timeout, 10)
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as error:
            raise OllamaError(
                f"Ollama est inaccessible sur {self.base_url}. "
                "Vérifiez que le service est démarré."
            ) from error
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise OllamaError("Réponse invalide de l'API Ollama /api/tags.")
        names = []
        for model in data["models"]:
            if not isinstance(model, dict):
                raise OllamaError("Modèle invalide dans la réponse Ollama.")
            name = model.get("name") or model.get("model")
            if isinstance(name, str):
                names.append(name)
        return sorted(names)

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        seed: int,
    ) -> tuple[dict[str, Any], float]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed},
        }
        started = perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data.get("response"), str):
                    raise OllamaError("Réponse Ollama sans champ texte 'response'.")
                return data, perf_counter() - started
            except (requests.RequestException, ValueError, OllamaError) as error:
                last_error = error
                if attempt < self.max_attempts:
                    sleep(min(2 ** (attempt - 1), 4))
        raise OllamaError(
            f"Échec Ollama après {self.max_attempts} tentative(s) : {last_error}"
        ) from last_error


def parse_ai_answer(raw_answer: str, choices: list[str]) -> str | None:
    """Convertit une réponse courte du modèle en lettre, sans matching flou."""

    valid_labels = string.ascii_uppercase[: len(choices)]
    cleaned = raw_answer.strip().strip("`*_\"' ")
    match = re.fullmatch(
        r"(?:(?:answer|réponse)\s*:\s*)?([A-Z])(?:[.)])?",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        label = match.group(1).upper()
        return label if label in valid_labels else None

    normalized = cleaned.casefold()
    matches = [
        string.ascii_uppercase[index]
        for index, choice in enumerate(choices)
        if choice.strip().casefold() == normalized
    ]
    return matches[0] if len(matches) == 1 else None


def benchmark_id(
    question_id: str,
    model: str,
    prompt_id: str,
    temperature: float,
    seed: int,
) -> str:
    identity = {
        "question_id": question_id,
        "model": model,
        "prompt_id": prompt_id,
        "prompt_template": PROMPT_TEMPLATES[prompt_id],
        "temperature": temperature,
        "seed": seed,
    }
    payload = json.dumps(
        identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Silver propre introuvable : {path}. Lancez d'abord build_silver."
        )
    table = pq.read_table(path)
    missing = set(QUESTION_COLUMNS) - set(table.column_names)
    if missing:
        raise ValueError(f"Colonnes absentes du Silver : {sorted(missing)}")
    return table.select(QUESTION_COLUMNS).to_pylist()


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    table = pq.read_table(path)
    if table.schema.remove_metadata() != RESULT_SCHEMA:
        raise ValueError(
            f"Schéma incompatible dans {path}. Utilisez une autre sortie ou "
            "migrez le fichier existant."
        )
    return {row["benchmark_id"]: row for row in table.to_pylist()}


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    metadata = {
        b"medallion_layer": b"silver",
        b"dataset_kind": b"llm_benchmark_results",
        b"silver_schema_version": b"1",
    }
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row["model_name"],
            row["prompt_id"],
            row["source_row_number"],
        ),
    )
    table = pa.Table.from_pylist(
        ordered_rows, schema=RESULT_SCHEMA.with_metadata(metadata)
    )
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
        if pq.ParquetFile(temporary_path).metadata.num_rows != len(rows):
            raise ValueError("Le checkpoint Parquet est incomplet.")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def nanoseconds_to_seconds(value: Any) -> float | None:
    return value / 1_000_000_000 if isinstance(value, int) else None


def result_from_response(
    question: dict[str, Any],
    model: str,
    prompt_id: str,
    prompt: str,
    temperature: float,
    seed: int,
    response: dict[str, Any] | None,
    response_time: float,
    error: Exception | None = None,
) -> dict[str, Any]:
    raw_answer = response.get("response") if response else None
    parsed_answer = (
        parse_ai_answer(raw_answer, question["answer_choices"])
        if isinstance(raw_answer, str)
        else None
    )
    status = "success" if parsed_answer is not None and error is None else "error"
    if error is None and parsed_answer is None:
        status = "invalid_answer"
    return {
        "benchmark_id": benchmark_id(
            question["question_id"], model, prompt_id, temperature, seed
        ),
        **{column: question[column] for column in QUESTION_COLUMNS},
        "model_name": model,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "generation_temperature": temperature,
        "generation_seed": seed,
        "ai_answer_raw": raw_answer,
        "ai_answer": parsed_answer,
        "ai_correct": parsed_answer == question["correct_choice"],
        "response_time": response_time,
        "status": status,
        "error_message": str(error) if error else None,
        "created_at": datetime.now(timezone.utc),
        "ollama_total_duration_seconds": nanoseconds_to_seconds(
            response.get("total_duration") if response else None
        ),
        "ollama_load_duration_seconds": nanoseconds_to_seconds(
            response.get("load_duration") if response else None
        ),
        "prompt_eval_count": response.get("prompt_eval_count") if response else None,
        "eval_count": response.get("eval_count") if response else None,
    }


def model_is_installed(model: str, installed: list[str]) -> bool:
    requested = model if ":" in model else f"{model}:latest"
    return model in installed or requested in installed


def run(args: argparse.Namespace, client: OllamaClient | None = None) -> int:
    questions = load_questions(args.input)
    questions = questions[args.offset :]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("Aucune question sélectionnée.")

    client = client or OllamaClient(
        args.ollama_url, args.timeout, args.max_attempts
    )
    installed = client.installed_models()
    if not model_is_installed(args.model, installed):
        available = ", ".join(installed) or "aucun"
        raise OllamaError(
            f"Le modèle {args.model!r} n'est pas installé. Modèles disponibles : "
            f"{available}."
        )

    results = load_existing_results(args.output)
    pending: list[tuple[dict[str, Any], str]] = []
    for question in questions:
        identifier = benchmark_id(
            question["question_id"],
            args.model,
            args.prompt_id,
            args.temperature,
            args.seed,
        )
        previous = results.get(identifier)
        if previous and previous["status"] == "success" and not args.force:
            continue
        pending.append((question, identifier))

    print(
        f"{len(questions)} question(s) sélectionnée(s), "
        f"{len(pending)} à exécuter, {len(questions) - len(pending)} déjà réussie(s)."
    )
    if not pending:
        return 0

    try:
        for index, (question, identifier) in enumerate(pending, start=1):
            prompt = render_prompt(
                question["question"], question["answer_choices"], args.prompt_id
            )
            started = perf_counter()
            try:
                response, response_time = client.generate(
                    args.model,
                    prompt,
                    args.temperature,
                    args.seed,
                )
                result = result_from_response(
                    question,
                    args.model,
                    args.prompt_id,
                    prompt,
                    args.temperature,
                    args.seed,
                    response,
                    response_time,
                )
            except OllamaError as error:
                result = result_from_response(
                    question,
                    args.model,
                    args.prompt_id,
                    prompt,
                    args.temperature,
                    args.seed,
                    None,
                    perf_counter() - started,
                    error,
                )
            results[identifier] = result
            print(
                f"[{index}/{len(pending)}] {question['question_id'][:8]} "
                f"{result['status']} {result['response_time']:.2f}s"
            )
            if index % args.checkpoint_every == 0:
                write_results(list(results.values()), args.output)
    except KeyboardInterrupt:
        write_results(list(results.values()), args.output)
        print(f"\nInterruption : progression enregistrée dans {args.output}.")
        return 130

    write_results(list(results.values()), args.output)
    print(f"Résultats Silver enregistrés : {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Nom du modèle Ollama")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prompt-id", choices=sorted(PROMPT_TEMPLATES), default="letter_only_v1"
    )
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--force", action="store_true", help="Rejoue aussi les résultats réussis"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit doit être strictement positif")
    if args.offset < 0:
        parser.error("--offset doit être positif ou nul")
    if args.timeout <= 0 or args.max_attempts <= 0 or args.checkpoint_every <= 0:
        parser.error("timeout, tentatives et fréquence de checkpoint doivent être positifs")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature doit être positive ou nulle")
    if not -(2**31) <= args.seed < 2**31:
        parser.error("--seed doit tenir dans un entier signé 32 bits")
    return args


def main() -> None:
    try:
        sys.exit(run(parse_args()))
    except (FileNotFoundError, ValueError, OllamaError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
