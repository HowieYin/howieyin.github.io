#!/usr/bin/env python3
"""Render latest-three homepage lists and full column archives."""

import json
import re
from datetime import date
from html import escape
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def load_catalog():
    columns = json.loads((ROOT / "columns.json").read_text(encoding="utf-8"))["columns"]
    slugs = [column["slug"] for column in columns]
    if len(set(slugs)) != len(slugs) or any(
        not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) for slug in slugs
    ):
        raise ValueError("Column slugs must be unique, lowercase URL segments.")
    paths = set()
    for column in columns:
        for article in column["articles"]:
            day = date.fromisoformat(article["date"])
            if article.get("periodStart") and date.fromisoformat(article["periodStart"]) > day:
                raise ValueError(f"Invalid coverage period: {article['title']}.")
            link = article["path"]
            parsed, path = urlsplit(link), PurePosixPath(link)
            if (
                parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                or "\\" in link or path.is_absolute() or ".." in path.parts
                or len(path.parts) < 2 or path.parts[0] != column["slug"]
                or path.name == "index.html" and len(path.parts) == 2
            ):
                raise ValueError(f"Invalid local article path: {link}.")
            if link in paths:
                raise ValueError(f"Duplicate article path: {link}.")
            paths.add(link)
            if not (ROOT / link).resolve().is_relative_to(ROOT.resolve()) or not (ROOT / link).is_file():
                raise ValueError(f"Missing article: {link}.")
        # Equal dates retain catalog order; prepend new entries.
        column["articles"] = sorted(column["articles"], key=lambda a: date.fromisoformat(a["date"]), reverse=True)
    return columns


def article_date(article):
    day = date.fromisoformat(article["date"])
    if article.get("periodStart"):
        start = date.fromisoformat(article["periodStart"])
        return (
            f'Coverage: <time datetime="{start}">{start}</time>'
            f' — <time datetime="{day}">{day}</time>'
        )
    return f'<time datetime="{day}">{day:%B} {day.day}, {day.year}</time>'


def home_section(column):
    slug = column["slug"]
    articles = column["articles"][:3]
    rows = []
    for index, article in enumerate(articles, 1):
        rows.append(f'''<li class="paper-latest-item">
<span class="paper-reading-index" aria-hidden="true">{index:02d}</span>
<div><a class="paper-latest-title" href="{escape(article["path"], quote=True)}">{escape(article["title"])} <span aria-hidden="true">↗</span></a>
<p class="paper-latest-meta">{article_date(article)} · {escape(article["topic"])}</p></div>
</li>''')
    latest = (
        f'<ol class="paper-latest-list" aria-labelledby="latest-{slug}-title">{"".join(rows)}</ol>'
        if rows else f'<p class="empty-text column-empty">{escape(column["emptyText"])}</p>'
    )
    return f'''<section class="content-section" id="{slug}" aria-labelledby="{slug}-title">
            <div class="section-heading">
              <span class="section-index" aria-hidden="true">{escape(column["index"])}</span>
              <div>
                <h2 id="{slug}-title">{escape(column["title"])}</h2>
                <p class="section-label">{escape(column["label"])}</p>
              </div>
            </div>
            <div class="section-body">
              <p class="section-description">{escape(column["description"])}</p>
              <h3 class="paper-subheading" id="latest-{slug}-title">{escape(column["latestLabel"])} <span>{len(articles):02d}</span></h3>
              {latest}
              <a class="text-link column-visit" href="{slug}/">Visit {escape(column["title"])} <span aria-hidden="true">↗</span></a>
            </div>
          </section>'''


def archive_section(column):
    entries = []
    for article in column["articles"]:
        link = escape(str(PurePosixPath(article["path"]).relative_to(column["slug"])), quote=True)
        entries.append(f'''<article class="paper-entry">
<p class="paper-metadata">{article_date(article)} · {escape(article["topic"])}</p>
<h3 class="paper-entry-title"><a href="{link}">{escape(article["title"])}</a></h3>
<p class="paper-entry-description">{escape(article["description"])}</p>
<a class="text-link" href="{link}">{escape(column["readLabel"])} <span aria-hidden="true">↗</span></a>
</article>''')
    body = "\n".join(entries) or f'<p class="empty-text column-empty">{escape(column["emptyText"])}</p>'
    count = len(entries)
    noun = "survey" if column["slug"] == "surveys" else "article"
    return f'''<section class="column-archive paper-category-archive" aria-labelledby="archive-title">
          <div class="paper-directory-heading">
            <h2 id="archive-title">{escape(column["archiveLabel"])}</h2>
            <p>{count} {noun}{"s" if count != 1 else ""} · Newest first</p>
          </div>
          {body}
        </section>'''


def replace_section(content, pattern, replacement, label):
    content, count = re.subn(pattern, lambda _: replacement, content, flags=re.S)
    if count != 1:
        raise ValueError(f"Expected exactly one {label} section.")
    return content


def main():
    columns = load_catalog()
    home_path = ROOT / "index.html"
    home = home_path.read_text(encoding="utf-8")
    outputs = {}
    for column in columns:
        slug = column["slug"]
        home = replace_section(
            home, rf'<section class="content-section" id="{slug}".*?</section>',
            home_section(column), f"homepage {slug}",
        )
        path = ROOT / slug / "index.html"
        archive = path.read_text(encoding="utf-8")
        outputs[path] = replace_section(
            archive,
            r'<section class="(?:archive-empty|column-archive paper-category-archive)" aria-labelledby="archive-title">.*?</section>',
            archive_section(column), f"{slug} archive",
        )
    # Validate every catalog entry and template before changing any output.
    outputs[home_path] = home
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    print("Rendered latest-three lists and complete archives for " + ", ".join(c["title"] for c in columns) + ".")


if __name__ == "__main__":
    main()
