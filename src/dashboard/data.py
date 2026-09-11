"""Accès DuckDB en lecture seule et indicateurs du dashboard.

Les agrégations partent des faits filtrés, jamais de moyennes déjà agrégées.
Le grain de configuration est conservé même si l'appelant fournit plusieurs runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_COLUMNS = ["prompt_id", "generation_temperature", "generation_seed"]
GROUP_COLUMNS = ["model_name", *CONFIG_COLUMNS]
REQUIRED_COLUMNS = {
    "benchmark_id", "question_id", *GROUP_COLUMNS, "category", "difficulty",
    "question_type", "question", "correct_answer", "correct_choice", "answer_choices",
    "ai_answer", "ai_answer_raw", "ai_correct", "status", "error_message",
    "response_time", "prompt_tokens", "completion_tokens", "total_tokens",
    "created_at", "gold_built_at",
}


class DashboardDataError(RuntimeError):
    """Problème de données à afficher avec une action de résolution."""


def database_path() -> Path:
    root = Path(os.environ.get("BI_BENCHMARK_ROOT", PROJECT_ROOT)).expanduser()
    return Path(os.environ.get("BI_BENCHMARK_DB", root / "data/gold/benchmark.duckdb")).expanduser().resolve()


def file_signature(path: Path) -> tuple:
    """Invalide le cache lors d'un build, y compris avec un journal WAL présent."""
    return tuple(
        (item.stat().st_mtime_ns, item.stat().st_size) if item.exists() else None
        for item in (path, Path(f"{path}.wal"))
    )


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DashboardDataError("La couche Gold n'existe pas encore. Lance `make dbt-build`, puis actualise cette page.")
    try:
        # Une connexion courte évite de maintenir un verrou entre deux interactions.
        with duckdb.connect(str(path), read_only=True) as connection:
            frame = connection.execute("select * from main.fct_benchmark_results").fetchdf()
    except duckdb.IOException as error:
        raise DashboardDataError(
            "La base Gold est indisponible ou verrouillée. Ferme la session DuckDB "
            "ouverte avec `.exit` (ou attends la fin de dbt), puis actualise cette page."
        ) from error
    except duckdb.CatalogException as error:
        raise DashboardDataError(
            "La table détaillée du dashboard manque dans la Gold. Lance `make dbt-build`, puis actualise cette page."
        ) from error
    if REQUIRED_COLUMNS - set(frame.columns):
        raise DashboardDataError("Le schéma Gold est ancien. Lance `make dbt-build`, puis actualise cette page.")
    for column in ("created_at", "gold_built_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def common_questions(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Intersection par configuration ; une erreur reste une tentative observée."""
    if not models:
        return frame.iloc[:0].copy()
    selected = frame.loc[frame["model_name"].isin(models)]
    keys = [*CONFIG_COLUMNS, "question_id"]
    coverage = selected.groupby(keys, dropna=False)["model_name"].transform("nunique")
    return selected.loc[coverage.eq(len(models))].copy()


def summarize(frame: pd.DataFrame, dimension: str | None = None) -> pd.DataFrame:
    keys = [*GROUP_COLUMNS, *([dimension] if dimension else [])]
    prepared = frame.assign(
        is_valid=frame["status"].eq("success"),
        is_invalid=frame["status"].eq("invalid_answer"),
        is_error=frame["status"].eq("error"),
    )
    result = prepared.groupby(keys, dropna=False, observed=True).agg(
        total_questions=("benchmark_id", "size"),
        correct_answers=("ai_correct", "sum"),
        valid_answers=("is_valid", "sum"),
        invalid_answers=("is_invalid", "sum"),
        technical_errors=("is_error", "sum"),
        avg_response_time=("response_time", "mean"),
        median_response_time=("response_time", "median"),
        p95_response_time=("response_time", lambda values: values.quantile(0.95)),
        avg_prompt_tokens=("prompt_tokens", "mean"),
        avg_completion_tokens=("completion_tokens", "mean"),
    ).reset_index()
    denominator = result["total_questions"].replace(0, float("nan"))
    result["overall_accuracy"] = 100 * result["correct_answers"] / denominator
    result["valid_answer_accuracy"] = 100 * result["correct_answers"] / result["valid_answers"].replace(0, float("nan"))
    for metric, count in (
        ("valid_answer_rate", "valid_answers"),
        ("invalid_answer_rate", "invalid_answers"),
        ("technical_error_rate", "technical_errors"),
    ):
        result[metric] = 100 * result[count] / denominator
    return result


def export_csv(frame: pd.DataFrame) -> bytes:
    """CSV UTF-8 pour Excel, avec neutralisation des formules dans les textes."""
    exported = frame.copy()
    for column in exported.select_dtypes(include=["object", "string"]).columns:
        exported[column] = exported[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
            else value
        )
    return exported.to_csv(index=False).encode("utf-8-sig")
