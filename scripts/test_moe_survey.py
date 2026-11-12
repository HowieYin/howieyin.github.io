#!/usr/bin/env python3
"""Publication regression for the MoE AI Surveys article."""

import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "surveys" / "moe-structure-exploration" / "index.html"
ARTICLE_DIGEST = "31605229aaeafeec011a151455eb3b965c43810e35374b930cd774c9b72de637"


class MoeSurveyPublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = ARTICLE.read_text(encoding="utf-8")

    def test_article_shell_and_version_binding(self):
        self.assertIn("<title>MoE结构探索：从条件计算到路由、专家结构与系统代价 — Howie Yin</title>", self.html)
        self.assertIn(f'<meta content="{ARTICLE_DIGEST}" name="article-sha256">', self.html)
        self.assertIn('rel="canonical"', self.html)
        self.assertIn('href="../../styles.css?v=3"', self.html)
        self.assertIn('href="../../assets/article-navigation.css"', self.html)
        self.assertIn('src="../../assets/article-navigation.js"', self.html)
        self.assertRegex(
            self.html,
            r'<main class="report" id="main">\s*<p class="eyebrow">AI SURVEYS / MIXTURE OF EXPERTS</p>\s*<details class="toc" hidden>',
        )
        self.assertIn('aria-current="page" href="../"', self.html)

    def test_mobile_contents_control_stays_in_flow(self):
        css = (ROOT / "assets" / "article-navigation.css").read_text(encoding="utf-8")
        script = (ROOT / "assets" / "article-navigation.js").read_text(encoding="utf-8")
        self.assertIn(".contents-toggle { position: static", css)
        self.assertIn("main.before(toggle)", script)
        self.assertNotIn("document.body.append(toggle, backdrop)", script)

    def test_public_article_has_no_internal_workflow_leakage(self):
        for forbidden in (
            "/Users/bytedance/",
            "runner/checks/",
            "awaits reviewer",
            "待审核",
            "不代表审核通过",
        ):
            self.assertNotIn(forbidden, self.html)

    def test_all_local_resources_resolve(self):
        refs = re.findall(r'(?:href|src)="([^"]+)"', self.html)
        missing = []
        for ref in refs:
            parsed = urlsplit(ref)
            if parsed.scheme or parsed.netloc or ref.startswith("#"):
                continue
            target = (ARTICLE.parent / parsed.path).resolve()
            if not target.is_relative_to(ROOT.resolve()) or not target.exists():
                missing.append(ref)
        self.assertEqual([], missing)

    def test_component_figures_and_readers_are_published(self):
        base = ARTICLE.parent
        self.assertTrue((base / "components" / "contract-explorer.html").is_file())
        figures = sorted((base / "figures").glob("*.svg"))
        readers = sorted((base / "figures" / "readers").glob("*.html"))
        self.assertEqual(7, len(figures))
        self.assertEqual(7, len(readers))
        for reader in readers:
            content = reader.read_text(encoding="utf-8")
            self.assertNotIn("/Users/bytedance/", content)
            self.assertRegex(content, r'\.\./[^"\']+\.svg')

    def test_catalog_entry_matches_public_article(self):
        catalog = json.loads((ROOT / "columns.json").read_text(encoding="utf-8"))
        surveys = next(column for column in catalog["columns"] if column["slug"] == "surveys")
        matches = [
            article for article in surveys["articles"]
            if article["path"] == "surveys/moe-structure-exploration/index.html"
        ]
        self.assertEqual(1, len(matches))
        self.assertEqual("2026-11-12", matches[0]["date"])
        self.assertEqual("MoE architecture & systems", matches[0]["topic"])


if __name__ == "__main__":
    unittest.main()
