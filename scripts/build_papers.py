#!/usr/bin/env python3
"""Render paper navigation from the catalog; no runtime JavaScript required."""

import json
import re
from datetime import date
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "papers/readings.json"


def load_catalog():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    categories, papers = catalog["categories"], catalog["papers"]
    for entries in (categories, papers):
        slugs = [entry["slug"] for entry in entries]
        if len(set(slugs)) != len(slugs) or any(not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) for slug in slugs):
            raise ValueError("Catalog slugs must be unique, lowercase URL segments.")
    category_slugs = {category["slug"] for category in categories}
    for paper in papers:
        if paper["category"] not in category_slugs:
            raise ValueError(f"Unknown category for {paper['slug']}.")
        date.fromisoformat(paper["readDate"])
        if not (ROOT / "papers" / paper["slug"] / "index.html").is_file():
            raise ValueError(f"Missing article: {paper['slug']}.")
    # Stable sort preserves editorial order when reading dates are equal.
    return categories, sorted(papers, key=lambda paper: date.fromisoformat(paper["readDate"]), reverse=True)


def reading_date(paper):
    day = date.fromisoformat(paper["readDate"])
    return f'<time datetime="{day.isoformat()}">{day:%B} {day.day}, {day.year}</time>'


def count_label(count):
    return f"{count} reading{'s' if count != 1 else ''}"


def topic_navigation(categories, papers, prefix, current=None):
    links = []
    for category in categories:
        count = sum(p["category"] == category["slug"] for p in papers)
        active = ' aria-current="page"' if category["slug"] == current else ""
        links.append(
            f'<li><a href="{prefix}{category["slug"]}/"{active}>'
            f'<span>{escape(category["title"])}</span>'
            f'<span class="paper-topic-count" aria-label="{count_label(count)}">{count}</span>'
            '<span class="paper-topic-arrow" aria-hidden="true">→</span></a></li>'
        )
    return '<ul class="paper-topic-links" aria-label="Paper reading topics">\n' + "\n".join(links) + "\n</ul>"


def article(paper, prefix, latest=False):
    label = '<p class="paper-latest-label">Latest reading</p>' if latest else ""
    return f'''<article class="paper-entry">
{label}
<p class="paper-metadata">{escape(paper["metadata"])}</p>
<h3 class="paper-entry-title"><a href="{prefix}{paper["slug"]}/">{escape(paper["title"])} <span aria-hidden="true">↗</span></a></h3>
<p class="paper-entry-description">{escape(paper["description"])}</p>
</article>'''


def page(title, description, body, root, canonical, archive=False):
    papers_link = "./" if archive else "../../"
    current = ' aria-current="page"' if archive else ' aria-current="location"'
    return f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#0c1016">
    <meta name="color-scheme" content="dark">
    <meta name="description" content="{escape(description, quote=True)}">
    <title>{escape(title)} — Howie Yin</title>
    <link rel="canonical" href="https://howieyin.github.io/{canonical}">
    <link rel="icon" href="{root}assets/favicon.svg" type="image/svg+xml">
    <link rel="stylesheet" href="{root}styles.css?v=3">
    <link rel="stylesheet" href="{root}papers/archive.css?v=1">
  </head>
  <body>
    <a class="skip-link" href="#main">Skip to content</a>
    <div class="page">
      <header class="site-header">
        <a class="wordmark" href="{root}" aria-label="Howie Yin, home"><span class="brand-mark" aria-hidden="true"></span>howieyin</a>
        <nav aria-label="Main navigation">
          <a href="{root}weekly/">Weekly</a>
          <a href="{root}foundations/">Foundations</a>
          <a href="{root}lab/">Lab</a>
          <a href="{papers_link}"{current}>Paper Readings</a>
          <a href="{root}#about">About</a>
        </nav>
      </header>
      <main id="main">
{body}
      </main>
      <footer class="site-footer">
        <span>© 2026 Howie Yin</span>
        <div class="footer-links"><a href="https://github.com/HowieYin/howieyin.github.io">Source <span aria-hidden="true">↗</span></a><a href="{root}">Back to home <span aria-hidden="true">←</span></a></div>
      </footer>
    </div>
  </body>
</html>
'''


def overview(categories, papers):
    rows = []
    for index, category in enumerate(categories, 1):
        readings = [p for p in papers if p["category"] == category["slug"]]
        preview = article(readings[0], "", latest=True) if readings else '<p class="empty-text">No readings yet.</p>'
        rows.append(f'''<section class="paper-topic-row" aria-labelledby="topic-{category["slug"]}">
<div class="paper-topic-heading">
  <p class="eyebrow">{index:02d} / {count_label(len(readings)).upper()}</p>
  <h3 id="topic-{category["slug"]}"><a href="topics/{category["slug"]}/">{escape(category["title"])}</a></h3>
  <p>{escape(category["description"])}</p>
</div>
<div class="paper-topic-preview">
  {preview}
  <a class="paper-more" href="topics/{category["slug"]}/" aria-label="More {escape(category["title"], quote=True)} readings">More <span aria-hidden="true">→</span></a>
</div>
</section>''')
    rows_html = "\n".join(rows)
    latest = f' · Last read {reading_date(papers[0])}' if papers else ""
    return f'''<section class="profile column-profile paper-profile" aria-labelledby="page-title">
  <p class="eyebrow">04 / THE PAPERS</p>
  <h1 id="page-title">Paper Readings</h1>
  <p class="profile-description column-intro">Read the paper. Understand the mechanism. Examine the evidence.</p>
  <p class="section-description column-intro">A personal record of research papers I have read, organized around the ideas behind them. Explore a topic to find its full reading archive.</p>
  <p class="paper-archive-status">{count_label(len(papers))} · {len(categories)} topics{latest}</p>
</section>
<section class="paper-directory" aria-labelledby="archive-title">
  <div class="paper-directory-heading"><h2 id="archive-title">Browse by topic</h2><p>One latest reading from each topic.</p></div>
  {rows_html}
</section>
<section class="content-section" aria-labelledby="approach-title">
  <div class="section-heading"><div><h2 id="approach-title">Inside a close reading</h2><p class="section-label">From claims to understanding</p></div></div>
  <div class="section-body">
    <ol class="reading-path">
      <li><strong>The source.</strong> Paper title, authors, publication and version, original link, code, and reading date.</li>
      <li><strong>The mechanism.</strong> Problem, prior approaches, core idea, equations, architecture, and original figures.</li>
      <li><strong>The evidence.</strong> Experimental setup, baselines, results, ablations, and limitations.</li>
      <li><strong>The takeaway.</strong> Personal insights, engineering relevance, and open questions, separate from the paper's claims.</li>
    </ol>
  </div>
</section>'''


def category_page(category, categories, papers):
    readings = [p for p in papers if p["category"] == category["slug"]]
    entries = "\n".join(article(paper, "../../") for paper in readings)
    title = escape(category["title"])
    return f'''<nav class="paper-breadcrumb" aria-label="Breadcrumb">
  <a href="../../../">Home</a><span aria-hidden="true">/</span><a href="../../">Paper Readings</a><span aria-hidden="true">/</span><span aria-current="page">{title}</span>
</nav>
<section class="profile column-profile paper-category-profile" aria-labelledby="page-title">
  <p class="eyebrow">PAPER READINGS / TOPIC</p>
  <h1 id="page-title">{title}</h1>
  <p class="profile-description column-intro">{escape(category["description"])}</p>
  <p class="paper-archive-status">{count_label(len(readings))} · Newest readings first</p>
</section>
<nav class="paper-category-navigation" aria-label="Browse other topics">
  {topic_navigation(categories, papers, "../", category["slug"])}
</nav>
<section class="paper-category-archive" aria-labelledby="readings-title">
  <div class="paper-directory-heading"><h2 id="readings-title">All readings</h2><a class="text-link" href="../../">All topics <span aria-hidden="true">←</span></a></div>
  {entries or '<p class="empty-text">No readings yet.</p>'}
</section>'''


def home_section(categories, papers):
    category_titles = {c["slug"]: c["title"] for c in categories}
    latest = []
    for index, paper in enumerate(papers[:3], 1):
        latest.append(f'''<li class="paper-latest-item">
<span class="paper-reading-index" aria-hidden="true">{index:02d}</span>
<div><a class="paper-latest-title" href="papers/{paper["slug"]}/">{escape(paper["title"])} <span aria-hidden="true">↗</span></a>
<p class="paper-latest-meta">{reading_date(paper)} · {escape(category_titles[paper["category"]])}</p></div>
</li>''')
    return f'''<section class="content-section" id="papers" aria-labelledby="papers-title">
            <div class="section-heading">
              <span class="section-index" aria-hidden="true">04</span>
              <div>
                <h2 id="papers-title">Paper Readings</h2>
                <p class="section-label">A personal research reading archive</p>
              </div>
            </div>
            <div class="section-body">
              <p class="section-description">A record of the papers I have read: the problems they address, how their methods work, what the experiments show, and what I take away for research and engineering.</p>
              <h3 class="paper-subheading" id="latest-readings-title">Latest readings <span>{min(3, len(papers)):02d}</span></h3>
              <ol class="paper-latest-list" aria-labelledby="latest-readings-title">{"".join(latest)}</ol>
              <div class="paper-home-topics">
                <h3 class="paper-subheading">Browse by topic</h3>
                {topic_navigation(categories, papers, "papers/topics/")}
              </div>
              <a class="text-link" href="papers/">Visit Paper Readings <span aria-hidden="true">↗</span></a>
            </div>
          </section>'''


def main():
    categories, papers = load_catalog()
    home_path = ROOT / "index.html"
    home = home_path.read_text(encoding="utf-8")
    home, replacements = re.subn(
        r'<section class="content-section" id="papers".*?</section>',
        lambda _: home_section(categories, papers), home, flags=re.S,
    )
    if replacements != 1:
        raise ValueError("Expected exactly one homepage Paper Readings section.")
    stylesheet = '    <link rel="stylesheet" href="papers/archive.css?v=1">\n'
    if stylesheet.strip() not in home:
        home = home.replace('    <link rel="stylesheet" href="styles.css?v=3">\n',
                            '    <link rel="stylesheet" href="styles.css?v=3">\n' + stylesheet)
    outputs = {home_path: home}
    outputs[ROOT / "papers/index.html"] = page(
        "Paper Readings", "Howie Yin's close readings of research papers, organized by model architecture, attention, long context, quantization, speculative decoding, and training.",
        overview(categories, papers), "../", "papers/", archive=True,
    )
    for category in categories:
        outputs[ROOT / "papers/topics" / category["slug"] / "index.html"] = page(
            f'{category["title"]} — Paper Readings', category["description"],
            category_page(category, categories, papers), "../../../", f'papers/topics/{category["slug"]}/',
        )
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"Rendered {len(papers)} readings across {len(categories)} topics; homepage shows {min(3, len(papers))}.")


if __name__ == "__main__":
    main()
