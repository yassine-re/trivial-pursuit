import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

from src.enrich.prompts import render_prompt
from src.enrich.run_benchmark import (
    parse_ai_answer,
    result_from_response,
    run,
    write_results,
)
from src.transform.build_silver import SILVER_SCHEMA


class FakeLMStudioClient:
    def __init__(self):
        self.generate_calls = 0

    def installed_models(self):
        return ["test-model"]

    def generate(self, model, prompt, temperature, seed):
        self.generate_calls += 1
        return {
            "choices": [{"message": {"content": "A"}}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 1,
                "total_tokens": 21,
            },
        }, 1.1


class BenchmarkTests(unittest.TestCase):
    def question(self):
        return {
            "source_row_number": 2,
            "question_id": "a" * 64,
            "category": "Science & Nature",
            "type": "multiple",
            "difficulty": "easy",
            "question": "What is H2O?",
            "correct_answer": "Water",
            "incorrect_answers": ["Gold", "Oxygen", "Salt"],
            "answer_choices": ["Gold", "Water", "Oxygen", "Salt"],
            "correct_choice": "B",
        }

    def test_prompt_never_contains_answer_metadata(self):
        question = self.question()
        prompt = render_prompt(
            question["question"], question["answer_choices"], "letter_only_v1"
        )
        self.assertIn("B. Water", prompt)
        self.assertNotIn("correct_choice", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_answer_parser_is_deliberately_strict(self):
        choices = self.question()["answer_choices"]
        self.assertEqual(parse_ai_answer("B", choices), "B")
        self.assertEqual(parse_ai_answer("Answer: B.", choices), "B")
        self.assertEqual(parse_ai_answer("Water", choices), "B")
        self.assertIsNone(parse_ai_answer("B because it is water", choices))
        self.assertIsNone(parse_ai_answer("Z", choices))

    def test_result_round_trip(self):
        question = self.question()
        prompt = render_prompt(
            question["question"], question["answer_choices"], "letter_only_v1"
        )
        result = result_from_response(
            question,
            "test-model",
            "letter_only_v1",
            prompt,
            0.0,
            42,
            {
                "choices": [{"message": {"content": "B"}}],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 1,
                    "total_tokens": 31,
                },
            },
            2.1,
        )
        self.assertTrue(result["ai_correct"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["prompt_tokens"], 30)
        self.assertEqual(result["completion_tokens"], 1)
        self.assertEqual(result["total_tokens"], 31)

        with tempfile.TemporaryDirectory(prefix="trivia-results-") as temporary:
            output = Path(temporary) / "results.parquet"
            write_results([result], output)
            loaded = pq.read_table(output).to_pylist()[0]
            self.assertEqual(loaded["benchmark_id"], result["benchmark_id"])
            self.assertTrue(loaded["ai_correct"])

    def test_runner_writes_and_resumes_successful_results(self):
        question = self.question()
        with tempfile.TemporaryDirectory(prefix="trivia-runner-") as temporary:
            folder = Path(temporary)
            questions_path = folder / "questions.parquet"
            output_path = folder / "results.parquet"
            pq.write_table(
                pa.Table.from_pylist([question], schema=SILVER_SCHEMA),
                questions_path,
            )
            args = Namespace(
                input=questions_path,
                output=output_path,
                offset=0,
                limit=None,
                lm_studio_url="http://unused/v1",
                timeout=10.0,
                max_attempts=1,
                model="test-model",
                prompt_id="letter_only_v1",
                temperature=0.0,
                seed=42,
                force=False,
                checkpoint_every=10,
            )
            client = FakeLMStudioClient()
            self.assertEqual(run(args, client), 0)
            self.assertEqual(client.generate_calls, 1)
            rows = pq.read_table(output_path).to_pylist()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "success")

            self.assertEqual(run(args, client), 0)
            self.assertEqual(client.generate_calls, 1)


if __name__ == "__main__":
    unittest.main()
