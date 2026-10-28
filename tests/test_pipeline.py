"""Run:  python -m unittest"""
import unittest

import numpy as np

from src import analysis, model
from src.clean import clean, parse_cost, parse_rating
from tests.fake_data import make_fake_raw


class TestParsing(unittest.TestCase):
    def test_parse_rating(self):
        self.assertEqual(parse_rating("4.1/5"), 4.1)
        self.assertEqual(parse_rating("3.9 /5"), 3.9)
        self.assertTrue(np.isnan(parse_rating("NEW")))
        self.assertTrue(np.isnan(parse_rating("-")))

    def test_parse_cost(self):
        self.assertEqual(parse_cost("1,200"), 1200)
        self.assertTrue(np.isnan(parse_cost(None)))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = clean(make_fake_raw())

    def test_duplicates_removed(self):
        self.assertEqual(len(self.df), 1500)
        self.assertFalse(self.df.duplicated(["name", "address"]).any())

    def test_types(self):
        self.assertEqual(self.df["online_order"].dtype, bool)
        self.assertTrue(self.df["rating"].between(1, 5).all() or self.df["rating"].isna().any())

    def test_analysis_functions_run(self):
        self.assertIn("restaurants", analysis.overview(self.df))
        self.assertLess(analysis.compare_groups(self.df, "book_table")["p_value"], 0.05)
        for fn in (analysis.by_area, analysis.top_cuisines, analysis.rating_by_price,
                   analysis.by_restaurant_type, analysis.opportunity_gaps):
            self.assertFalse(fn(self.df).empty, fn.__name__)

    def test_model_trains(self):
        result = model.train(self.df)
        self.assertGreater(result["roc_auc"], 0.5)
        self.assertNotIn("votes", " ".join(result["importance"].index))


if __name__ == "__main__":
    unittest.main()
