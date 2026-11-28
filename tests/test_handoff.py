"""Числа отчёта должны приходить из таблиц, а не из текста шаблона.

История, ради которой эти тесты написаны: в `reports/build_final_report.py` стояло «0 из 53» и «2 399
закрытых сделок». Данные уехали (в каталоге 54 ID, в журнале 2 035 закрытий), и отчёт начал противоречить
собственным таблицам — модель, которой поручили «поправить отчёт», потратила на это весь вечер.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "reports"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import handoff  # noqa: E402
import report_facts as rf  # noqa: E402

LAB = """id,family,hypothesis,unit,trades,clusters,open,mean_net,t,holm,hit,total_net,mean_gross,dsr,verdict
D01,D,reversal,return,10,10,1,-0.0018,-1.06,1.0,0.4,-0.018,-0.0008,0.0003,nothing
D02,D,long,return,0,0,0,,,,,0.0,—,—,no signal yet
D03,D,short,return,6,6,0,0.0020,2.40,0.010,0.5,0.012,0.001,0.5,collected
"""
LEDGER = """event,ts,id,hyp,family,venue,instrument,side
OPEN,1,A11,mono,A,pm,x,y
CLOSE,2,A11,mono,A,pm,x,y
CLOSE,3,A11,mono,A,pm,x,y
OPEN,4,A12,spread,A,ks,x,y
"""
CATALOG = """| ID | Strategy |
|---|---|
| A11 | mono |
| A12 | spread |
| A13 | third |
"""


class FactsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "lab.csv").write_text(LAB, encoding="utf-8")
        (d / "ledger.csv").write_text(LEDGER, encoding="utf-8")
        (d / "STRATEGIES.md").write_text(CATALOG, encoding="utf-8")
        self.facts = rf.compute(d / "lab.csv", d / "ledger.csv", d / "STRATEGIES.md")

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts_come_from_files(self):
        self.assertEqual(self.facts["n_hypotheses"], 3)
        self.assertEqual(self.facts["n_with_trades"], 2)          # D01 и D03: >= 5 сделок
        self.assertEqual(self.facts["n_holm_positive"], 1)        # D03: holm 0.010 и плюс
        self.assertEqual(self.facts["n_catalog_ids"], 3)
        self.assertEqual(self.facts["n_catalog_active"], 2)       # D01 и D03 торговали
        self.assertEqual(self.facts["n_closed_trades"], 2)
        self.assertEqual(self.facts["n_open_trades"], 2)

    def test_missing_files_give_dash_not_zero(self):
        """Нет файла — значит «неизвестно» («—»), а не уверенный ноль: иначе отчёт врёт молча."""
        empty = Path(self.tmp.name) / "nope"
        facts = rf.compute(empty / "lab.csv", empty / "ledger.csv", empty / "STRATEGIES.md")
        self.assertTrue(all(v is None for v in facts.values()))
        self.assertEqual(rf.fmt(None), rf.NOT_COMPUTED)

    def test_substitute_fills_every_marker(self):
        text = "прошли __N_HOLM_POSITIVE__ из __N_HYPOTHESES__; сделок __N_CLOSED_TRADES__"
        out = rf.substitute(text, self.facts)
        self.assertEqual(out, "прошли 1 из 3; сделок 2")
        self.assertNotIn("__", out)

    def test_substitute_fails_loudly_on_unknown_marker(self):
        with self.assertRaises(rf.TemplateError):
            rf.substitute("__N_MYSTERY__", self.facts)

    def test_thousands_are_spaced_like_in_the_report(self):
        self.assertEqual(rf.fmt(2035), "2 035")


class TemplateTest(unittest.TestCase):
    """Шаблон отчёта не должен снова обрасти числами, которые нельзя проверить."""

    def setUp(self):
        self.src = (ROOT / "reports" / "build_final_report.py").read_text(encoding="utf-8")

    def test_no_stale_headline_literals(self):
        """Ищем литералы в коде, а не в комментариях: в комментарии эти числа — объяснение истории."""
        code = "\n".join(ln for ln in self.src.splitlines() if not ln.lstrip().startswith("#"))
        for bad in ("0<small> из 53</small>", "2 399", "из 53 торговых гипотез", "19 активных"):
            self.assertNotIn(bad, code, f"в шаблоне снова литерал: {bad}")

    def test_headline_numbers_are_markers(self):
        self.assertIn("__N_HOLM_POSITIVE__", self.src)
        self.assertIn("__N_CLOSED_TRADES__", self.src)

    def test_generator_fails_on_leftover_marker(self):
        self.assertIn("_facts_substitute(out, FACTS)", self.src)


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.facts = {"n_hypotheses": 46, "n_with_trades": 15, "n_holm_positive": 0,
                      "n_catalog_ids": 54, "n_catalog_active": 23, "n_closed_trades": 2035}

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, md: str, html: str = "") -> tuple[Path, Path]:
        md_path, html_path = self.dir / "r.md", self.dir / "r.html"
        md_path.write_text(md, encoding="utf-8")
        html_path.write_text(html, encoding="utf-8")
        return md_path, html_path

    def test_good_report_passes(self):
        md = "0 из 46 · 2 035 · 54 · 23 · 15"
        md_path, html_path = self._write(md, '<div class="v">0<small> из 46</small>')
        res = handoff.audit(md_path, html_path, self.facts)
        self.assertTrue(res["ok"], res["issues"])

    def test_wrong_kpi_is_caught(self):
        md = "0 из 46 · 2 035"
        md_path, html_path = self._write(md, '<div class="v">0<small> из 53</small>')
        res = handoff.audit(md_path, html_path, self.facts)
        self.assertFalse(res["ok"])
        self.assertTrue(any("из 53" in x for x in res["issues"]))

    def test_missing_closed_trades_count_is_caught(self):
        md_path, html_path = self._write("0 из 46")
        res = handoff.audit(md_path, html_path, self.facts)
        self.assertTrue(any("n_closed_trades" in x for x in res["issues"]))

    def test_leftover_marker_is_caught(self):
        md_path, html_path = self._write("0 из 46 · 2 035 · __N_CATALOG_IDS__")
        res = handoff.audit(md_path, html_path, self.facts)
        self.assertTrue(any("маркеры" in x for x in res["issues"]))


class HandoffFilesTest(unittest.TestCase):
    def test_handoff_written_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            md, js = Path(tmp) / "HANDOFF.md", Path(tmp) / "handoff.json"
            old_md, old_js = handoff.HANDOFF_MD, handoff.HANDOFF_JSON
            handoff.HANDOFF_MD, handoff.HANDOFF_JSON = md, js
            try:
                handoff.write_handoff(
                    {"version": "abc1234", "python": "3.14.0", "pandas": "3.1.1", "imports": {},
                     "inputs": {"data/x": {"what": "данные", "exists": True, "files": 3}},
                     "coverage": {"total": 2, "with_result": 1, "missing_min": 1.5,
                                  "missing": [{"name": "wings", "group": "крылья", "what": "правило",
                                               "est_min": 90}], "have": []}},
                    {"studies": 2, "ok": 1, "skipped": 1, "failed": [], "seconds": 12.0, "summary": ""},
                    {"ran": True, "rc": 0, "seconds": 1.0, "tail": ["done"]},
                    {"facts": self_facts(), "issues": [], "checked": 6, "ok": True})
            finally:
                handoff.HANDOFF_MD, handoff.HANDOFF_JSON = old_md, old_js
            text = md.read_text(encoding="utf-8")
            self.assertIn("Что ещё не посчитано: 1 из 2", text)
            self.assertIn("wings", text)
            self.assertIn("Аудит: **пройден**", text)
            self.assertIn("2 035", text)


def self_facts() -> dict:
    return {"n_hypotheses": 46, "n_with_trades": 15, "n_holm_positive": 0,
            "n_catalog_ids": 54, "n_catalog_active": 23, "n_closed_trades": 2035}


if __name__ == "__main__":
    unittest.main()
