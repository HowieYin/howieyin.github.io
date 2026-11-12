#!/usr/bin/env python3
"""Publish the reviewed MoE course artifact into AI Surveys."""

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "surveys" / "moe-structure-exploration"
ARTICLE_DIGEST = "31605229aaeafeec011a151455eb3b965c43810e35374b930cd774c9b72de637"
TITLE = "MoE结构探索：从条件计算到路由、专家结构与系统代价"


def replace_once(content: str, old: str, new: str, label: str) -> str:
    count = content.count(old)
    if count != 1:
        raise ValueError(f"Expected one {label}, found {count}.")
    return content.replace(old, new, 1)


def adapt_article(source: Path) -> str:
    content = (source / "report.html").read_text(encoding="utf-8")
    content = replace_once(
        content,
        f"<title>MoE结构探索：机制、证据与条件化设计</title>",
        (
            f"<title>{TITLE} — Howie Yin</title>"
            '<meta content="从概率混合、稀疏路由与Soft MoE，到专家结构、执行组织和系统代价：'
            '用四层保持性合同比较MoE机制、证据边界与条件化选型。" name="description">'
            f'<meta content="{ARTICLE_DIGEST}" name="article-sha256">'
            '<meta content="#0c1016" name="theme-color">'
            '<link href="https://howieyin.github.io/surveys/moe-structure-exploration/" rel="canonical">'
            '<link href="../../assets/favicon.svg" rel="icon" type="image/svg+xml">'
            '<link href="../../styles.css?v=3" rel="stylesheet">'
            '<link href="../../assets/article-navigation.css" rel="stylesheet">'
            '<script defer src="../../assets/article-navigation.js"></script>'
        ),
        "document title",
    )
    content = replace_once(
        content,
        'body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.85 system-ui,-apple-system,"PingFang SC",sans-serif;overflow-wrap:anywhere}',
        'body{margin:0;background:var(--background);color:var(--ink);font:17px/1.85 var(--sans);overflow-wrap:anywhere}',
        "body typography",
    )
    old_shell = (
        '</style></head><body><a class="skip" href="#main">Skip to article</a>'
        '<nav class="site-nav" aria-label="Site navigation">'
        '<a href="https://howieyin.github.io/">Home</a>'
        '<a href="https://howieyin.github.io/surveys/">AI Surveys</a>'
        '<a href="components/contract-explorer.html">Interactive Explorer</a></nav>'
        '<div class="shell"><aside aria-label="Article contents"><details open><summary>Contents</summary>'
    )
    new_shell = (
        '</style></head><body><a class="skip-link" href="#main">Skip to content</a>'
        '<div class="page"><header class="site-header">'
        '<a aria-label="Howie Yin, home" class="wordmark" href="../../">'
        '<span aria-hidden="true" class="brand-mark"></span>howieyin</a>'
        '<nav aria-label="Main navigation"><a href="../../weekly/">Weekly</a>'
        '<a href="../../foundations/">Foundations</a><a href="../../lab/">Lab</a>'
        '<a aria-current="page" href="../">Surveys</a><a href="../../#about">About</a></nav>'
        '</header><main class="report" id="main">'
        '<p class="eyebrow">AI SURVEYS / MIXTURE OF EXPERTS</p>'
        '<details class="toc" hidden><summary>Contents</summary>'
    )
    content = replace_once(content, old_shell, new_shell, "article shell")
    content = replace_once(
        content,
        '</ul></details></aside><main id="main" class="report"><h1>',
        '</ul></details><h1>',
        "table-of-contents boundary",
    )
    return content


def update_catalog() -> None:
    path = ROOT / "columns.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    surveys = next(column for column in catalog["columns"] if column["slug"] == "surveys")
    article = {
        "title": TITLE,
        "path": "surveys/moe-structure-exploration/index.html",
        "date": "2026-11-12",
        "description": (
            "A preservation-based comparison of probabilistic mixtures, sparse routing, "
            "Expert Choice, Soft MoE, expert structure and systems costs, with reproducible "
            "counterexamples and explicit evidence boundaries."
        ),
        "topic": "MoE architecture & systems",
    }
    surveys["articles"] = [
        article,
        *[item for item in surveys["articles"] if item["path"] != article["path"]],
    ]
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course_dir", type=Path)
    args = parser.parse_args()
    source = args.course_dir.expanduser().resolve()
    if not (source / "report.html").is_file():
        raise SystemExit(f"Missing reviewed report: {source / 'report.html'}")

    TARGET.mkdir(parents=True, exist_ok=True)
    for directory in ("components", "figures"):
        target = TARGET / directory
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source / directory, target)
    (TARGET / "index.html").write_text(adapt_article(source), encoding="utf-8")
    update_catalog()
    print(f"Published reviewed MoE bundle to {TARGET}")


if __name__ == "__main__":
    main()
