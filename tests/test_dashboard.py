"""Vérifie les dénominateurs, le périmètre commun et les interactions Streamlit."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.dashboard.data import (
    PROJECT_ROOT, DashboardDataError, common_questions, export_csv, load_results, summarize,
)


def fixtures():
    records = []
    for model, qid, status, correct, seed, prompt in (
        ("vendor/a", "q1", "success", True, 42, "p1"),
        ("vendor/a", "q2", "success", False, 42, "p1"),
        ("vendor/a", "q3", "invalid_answer", False, 42, "p1"),
        ("vendor/a", "q4", "error", False, 42, "p1"),
        ("vendor/b", "q1", "success", True, 42, "p1"),
        ("vendor/b", "q2", "error", False, 42, "p1"),
        ("vendor/a", "q1", "error", False, 43, "p1"),
        ("vendor/b", "q1", "success", True, 42, "p2"),
    ):
        records.append({
            "benchmark_id": f"{model}-{qid}-{seed}-{prompt}", "question_id": qid,
            "model_name": model, "prompt_id": prompt, "generation_temperature": 0.0,
            "generation_seed": seed, "category": "History" if qid == "q1" else "Science",
            "difficulty": "easy" if qid == "q1" else "hard", "question_type": "boolean",
            "question": f"Question {qid}", "correct_answer": "True", "correct_choice": "A",
            "answer_choices": ["True", "False"], "ai_answer": "A" if correct else None,
            "ai_answer_raw": "A" if correct else "unknown", "ai_correct": correct,
            "status": status, "error_message": "Test error" if status == "error" else None,
            "response_time": 1.0 if qid == "q1" else 3.0,
            "prompt_tokens": 10 if status == "success" else None,
            "completion_tokens": 1 if status == "success" else None,
            "total_tokens": 11 if status == "success" else None,
            "created_at": pd.Timestamp("2026-01-01", tz="UTC"),
            "gold_built_at": pd.Timestamp.now(tz="UTC"),
        })
    return pd.DataFrame(records)


class DashboardMetricsTests(unittest.TestCase):
    def test_configuration_grain_and_accuracy_denominators(self):
        results = summarize(fixtures())
        self.assertEqual(len(results), 4)
        selected = results.loc[(results.model_name == "vendor/a") & (results.generation_seed == 42)].iloc[0]
        self.assertEqual(selected.total_questions, 4)
        self.assertEqual(selected.overall_accuracy, 25)
        self.assertEqual(selected.valid_answer_accuracy, 50)
        self.assertEqual(selected.technical_error_rate, 25)
        self.assertEqual(selected.invalid_answer_rate, 25)
        self.assertEqual(selected.avg_prompt_tokens, 10)
        no_valid = results.loc[results.generation_seed == 43].iloc[0]
        self.assertEqual(no_valid.overall_accuracy, 0)
        self.assertTrue(pd.isna(no_valid.valid_answer_accuracy))

    def test_common_questions_does_not_drop_errors_or_cross_configurations(self):
        selected = common_questions(fixtures(), ["vendor/a", "vendor/b"])
        self.assertEqual(len(selected), 4)
        self.assertEqual(set(selected.question_id), {"q1", "q2"})
        self.assertEqual(set(selected.generation_seed), {42})
        self.assertEqual(set(selected.prompt_id), {"p1"})
        self.assertEqual(selected.status.eq("error").sum(), 1)
        self.assertTrue(common_questions(fixtures(), ["vendor/a", "missing"]).empty)

    def test_temperatures_are_separate_groups(self):
        frame = fixtures().iloc[:1].copy()
        variant = frame.assign(generation_temperature=0.7, ai_correct=False)
        summary = summarize(pd.concat([frame, variant], ignore_index=True))
        self.assertEqual(summary.overall_accuracy.tolist(), [100, 0])

    def test_unbalanced_categories_are_not_averaged_equally(self):
        frame = fixtures()
        frame = frame.loc[(frame.model_name == "vendor/a") & (frame.generation_seed == 42)]
        self.assertEqual(summarize(frame, "category").overall_accuracy.mean(), 50)
        self.assertEqual(summarize(frame).overall_accuracy.iloc[0], 25)

    def test_export_preserves_records_and_escapes_formulas(self):
        exported = export_csv(pd.DataFrame({"question": ["=1+1", "échantillon"], "accuracy": [25, 50]})).decode("utf-8-sig")
        self.assertIn("'=1+1", exported)
        self.assertIn("échantillon", exported)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absent.duckdb"
            with self.assertRaises(DashboardDataError):
                load_results(path)
            self.assertFalse(path.exists())


class DashboardAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trivia-dashboard-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "fixture.duckdb"
        with duckdb.connect(str(self.path)) as connection:
            connection.register("fixtures", fixtures())
            connection.execute("create table fct_benchmark_results as select * from fixtures")
        self.environment = patch.dict(os.environ, {"BI_BENCHMARK_DB": str(self.path)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=20)

    def test_load_filters_metrics_and_empty_search(self):
        app = self.app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        self.assertEqual(len(app.get("plotly_chart")), 4)
        # Le périmètre commun contient deux questions par modèle.
        self.assertEqual(app.dataframe[0].value.total_questions.tolist(), [2, 2])
        app.toggle(key="paired").set_value(False).run()
        self.assertEqual(sorted(app.dataframe[0].value.total_questions.tolist()), [2, 4])
        app.multiselect(key="categories").set_value(["History"]).run()
        self.assertTrue(app.dataframe[0].value.overall_accuracy.eq(100).all())
        app.radio(key="dimension").set_value("Difficulté").run()
        app.radio(key="metric").set_value("Sur réponses valides").run()
        app.text_input(key="search").set_value("no matching question").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Aucune réponse" in message.value for message in app.info))

    def test_empty_models_and_zero_valid_answers(self):
        app = self.app.run()
        app.multiselect(key="models").set_value([]).run()
        self.assertTrue(any("au moins un modèle" in message.value for message in app.info))
        app.multiselect(key="models").set_value(["vendor/a"]).run()
        app.selectbox(key="configuration").set_value(("p1", 0.0, 43)).run()
        app.radio(key="metric").set_value("Sur réponses valides").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.dataframe[0].value.valid_answer_accuracy.isna().all())

    def test_missing_table_has_actionable_message(self):
        with duckdb.connect(str(self.path)) as connection:
            connection.execute("drop table fct_benchmark_results")
        app = self.app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("make dbt-build" in message.value for message in app.error))


if __name__ == "__main__":
    unittest.main()
