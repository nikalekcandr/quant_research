import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.lit import bibliography as bib
from src.lit.import_openalex import merge, to_row


def row(**kw):
    base = dict(id="", domain="ml-methods", year=2020, authors="Doe, J.", title="A paper", venue="Journal",
                why="test", status="candidate", confidence="high", doi="")
    base.update(kw)
    return base


class StoreTest(unittest.TestCase):
    def test_shipped_store_loads_and_validates(self):
        rows = bib.load()
        self.assertGreater(len(rows), 150)
        errors, _ = bib.validate(rows)
        self.assertEqual(errors, [])

    def test_roundtrip_keeps_rows_and_ids(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "b.csv"
            bib.save([row(title="First"), row(title="Second", domain="crypto")], path)
            back = bib.load(path)
            self.assertEqual(len(back), 2)
            self.assertEqual([r["id"] for r in back], ["L001", "L002"])
            self.assertEqual({r["title"] for r in back}, {"First", "Second"})


class ValidateTest(unittest.TestCase):
    def test_missing_year_is_an_error(self):
        errors, _ = bib.validate([row(year=None)])
        self.assertTrue(any("год" in e for e in errors))

    def test_unknown_domain_is_an_error(self):
        errors, _ = bib.validate([row(domain="astrology")])
        self.assertTrue(any("домен" in e for e in errors))

    def test_duplicate_title_is_a_warning(self):
        errors, warnings = bib.validate([row(title="Same thing", id="L001"), row(title="same  THING!", id="L002")])
        self.assertEqual(errors, [])
        self.assertTrue(any("дубликат" in w for w in warnings))

    def test_duplicate_doi_is_a_warning(self):
        _, warnings = bib.validate([row(title="A", doi="10.1/x"), row(title="B", doi="10.1/X")])
        self.assertTrue(any("DOI" in w for w in warnings))


class DomainTest(unittest.TestCase):
    def test_keywords_pick_the_domain(self):
        cases = {"Deflated Sharpe ratio and backtest overfitting": "backtest-overfitting",
                 "Prediction markets and favourite-longshot bias": "prediction-markets",
                 "Deep learning for limit order books": "hf-econometrics",
                 "Realized volatility forecasting with HAR": "vol-forecasting",
                 "Bitcoin perpetual futures and funding": "crypto",
                 "Strictly proper scoring rules and calibration": "probability-scoring"}
        for title, want in cases.items():
            self.assertEqual(bib.guess_domain(title), want, title)

    def test_normalize_title_is_aggressive(self):
        self.assertEqual(bib.normalize_title("Volatility is Rough!"), bib.normalize_title("volatility  is rough"))


class ImportTest(unittest.TestCase):
    rec = {"title": "Machine learning for order books", "publication_year": 2021,
           "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
           "doi": "https://doi.org/10.1093/x/1",
           "primary_location": {"source": {"display_name": "Journal of Tests"}},
           "concepts": [{"display_name": "Limit order book"}]}

    def test_mapping_reads_venue_authors_doi(self):
        r = to_row(self.rec)
        self.assertEqual(r["venue"], "Journal of Tests")
        self.assertEqual(r["authors"], "Lovelace, A.")
        self.assertEqual(r["doi"], "10.1093/x/1")
        self.assertEqual(r["status"], "imported")
        self.assertEqual(r["domain"], "hf-econometrics")

    def test_record_without_year_is_skipped(self):
        self.assertIsNone(to_row({"title": "no year"}))

    def test_merge_dedupes_by_doi_and_title(self):
        existing = [row(title="Machine learning for order books", doi="")]
        added, n_add, n_skip = merge(existing, [to_row(self.rec), to_row(self.rec)])
        self.assertEqual(len(added), 1)          # first survives by title, second is a duplicate
        self.assertEqual((n_add, n_skip), (0, 2))


if __name__ == "__main__":
    unittest.main()
