import importlib
import math
import os
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import ingest

DATASET = Path(__file__).resolve().parents[1] / "imdb_top_1000.csv"


class IngestImportTests(unittest.TestCase):
    def test_import_has_no_side_effects(self):
        # Importing must not need credentials, a cluster, or OpenAI.
        try:
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "openai.OpenAI"
            ) as openai, mock.patch("couchbase.cluster.Cluster") as cluster, mock.patch(
                "dotenv.load_dotenv"
            ) as load_dotenv:
                importlib.reload(ingest)
        finally:
            importlib.reload(ingest)

        openai.assert_not_called()
        cluster.assert_not_called()
        load_dotenv.assert_not_called()

    def test_main_requires_environment_before_connecting(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            ingest, "load_dotenv"
        ), mock.patch.object(ingest, "connect_to_couchbase") as connect:
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                ingest.main()

        connect.assert_not_called()


class PrepareMovieRecordsTests(unittest.TestCase):
    def test_parses_gross_and_fills_missing_values(self):
        data = pd.DataFrame(
            {
                "Series_Title": ["A", "B"],
                "Gross": ["28,341,469", None],
                "Certificate": ["A", None],
                "Meta_score": [80.0, None],
            }
        )

        records = ingest.prepare_movie_records(data)

        self.assertEqual(records[0]["Gross"], 28341469.0)
        self.assertEqual(records[1]["Gross"], 0)
        self.assertEqual(records[1]["Certificate"], "NA")
        self.assertEqual(records[1]["Meta_score"], -1)
        # The caller's dataframe is left untouched.
        self.assertEqual(data.loc[0, "Gross"], "28,341,469")

    def test_bundled_dataset_produces_json_ready_records(self):
        records = ingest.prepare_movie_records(pd.read_csv(DATASET))

        self.assertEqual(len(records), 1000)
        for record in records:
            for field, value in record.items():
                if isinstance(value, float):
                    self.assertFalse(
                        math.isnan(value), f"{record['Series_Title']}.{field} is NaN"
                    )
            # Fields the search app filters and renders on.
            self.assertIsInstance(record["Released_Year"], int)
            self.assertIsInstance(record["IMDB_Rating"], float)
            self.assertIsInstance(record["Overview"], str)


if __name__ == "__main__":
    unittest.main()
