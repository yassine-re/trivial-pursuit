import csv
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from src.transform.build_silver import (
    DataQualityError,
    build_silver,
    transform_csv,
)


HEADER = [
    "category",
    "type",
    "difficulty",
    "question",
    "correct_answer",
    "incorrect_answers",
]


class SilverTransformationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="trivia-silver-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def write_csv(self, rows):
        path = self.folder / "questions.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def fixtures(self):
        return [
            {
                "category": "Science &amp; Nature",
                "type": "multiple",
                "difficulty": "easy",
                "question": " Who said &quot;Eureka&quot;? ",
                "correct_answer": "Archimedes",
                "incorrect_answers": json.dumps(["Newton", "Einstein", "Galileo"]),
            },
            {
                "category": "History",
                "type": "boolean",
                "difficulty": "medium",
                "question": "The test works.",
                "correct_answer": "True",
                "incorrect_answers": json.dumps(["False"]),
            },
        ]

    def test_cleaning_ids_choices_and_parquet_metadata(self):
        source = self.write_csv(self.fixtures())
        output = self.folder / "silver" / "questions.parquet"
        stats = build_silver(source, output)

        self.assertEqual(stats.source_rows, 2)
        self.assertEqual(stats.output_rows, 2)
        # La statistique compte les champs modifiés, pas chaque opération de
        # nettoyage appliquée à un même champ.
        self.assertGreaterEqual(stats.cleaned_values, 2)
        table = pq.read_table(output)
        rows = table.to_pylist()
        self.assertEqual(rows[0]["category"], "Science & Nature")
        self.assertEqual(rows[0]["question"], 'Who said "Eureka"?')
        self.assertEqual(len(rows[0]["question_id"]), 64)
        self.assertCountEqual(
            rows[0]["answer_choices"],
            ["Archimedes", "Newton", "Einstein", "Galileo"],
        )
        correct_index = ord(rows[0]["correct_choice"]) - ord("A")
        self.assertEqual(rows[0]["answer_choices"][correct_index], "Archimedes")
        metadata = table.schema.metadata
        self.assertEqual(metadata[b"medallion_layer"], b"silver")
        self.assertEqual(metadata[b"silver_schema_version"], b"1")

    def test_transformation_is_independent_from_csv_order(self):
        first = self.write_csv(self.fixtures())
        first_rows, _ = transform_csv(first)
        second = self.write_csv(list(reversed(self.fixtures())))
        second_rows, _ = transform_csv(second)
        first_by_id = {row["question_id"]: row for row in first_rows}
        second_by_id = {row["question_id"]: row for row in second_rows}
        self.assertEqual(set(first_by_id), set(second_by_id))
        for question_id in first_by_id:
            self.assertEqual(
                first_by_id[question_id]["answer_choices"],
                second_by_id[question_id]["answer_choices"],
            )

    def test_exact_duplicates_are_removed(self):
        row = self.fixtures()[0]
        source = self.write_csv([row, row])
        rows, stats = transform_csv(source)
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats.exact_duplicates_removed, 1)

    def test_invalid_number_of_answers_is_rejected(self):
        row = self.fixtures()[0]
        row["incorrect_answers"] = json.dumps(["Newton"])
        source = self.write_csv([row])
        with self.assertRaisesRegex(DataQualityError, "exige 3"):
            transform_csv(source)


if __name__ == "__main__":
    unittest.main()
