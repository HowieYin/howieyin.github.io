# Howie Yin

Personal website: **[https://howieyin.github.io/](https://howieyin.github.io/)**

A minimal personal homepage with a dark background, sans-serif typography,
monospace details and muted blue links. All navigation and interface copy is in
English. Column labels and navigation must remain in English; article titles
and bodies may use the language appropriate to their content.

## Columns

- **AI Weekly** (`weekly/`): AI industry reports covering models, research,
  hardware, infrastructure, agents, products and open source.
- **AI Foundations** (`foundations/`): fundamentals, technical deep dives,
  the evolution of ideas and worked examples.
- **AI Lab** (`lab/`): practical tricks, reproducible experiments,
  counterintuitive findings and myth testing.
- **Paper Readings** (`papers/`): a personal archive of papers read, with original
  sources, reading dates, methods, experiments, limitations and personal insights.
  Do not automatically count news coverage or knowledge articles as completed
  paper readings. Each reading should have its own page and archive entry.
- **AI Surveys** (`surveys/`): personal explorations of technical fields and
  research directions, comparing approaches, tracing their evolution and
  identifying open questions.

Each column has a homepage section and a standalone archive page. The homepage
shows up to three recent articles per column and a Visit link to the full archive.
AI Surveys is empty until the first real survey is published.

## Editing

- `index.html`: English homepage, column cards, About and Projects.
- `weekly/index.html`, `foundations/index.html`, `lab/index.html`, `papers/index.html`,
  `surveys/index.html`: column landing pages.
- `columns.json`: article catalog and homepage copy for Weekly, Foundations, Lab
  and Surveys.
- `styles.css`: layout, colors and responsive styles.
- `assets/`: locally served fonts and favicon. No external services are needed at
runtime.

After adding a static article inside `weekly/`, `foundations/`, `lab/` or
`surveys/`, prepend an entry to its column in `columns.json` and run:

```bash
python3 scripts/build_columns.py
```

Each article has a `title`, repository-relative `path`, ISO `date`, `description`
and English `topic`. For weekly reports, `date` is the coverage end date and
`periodStart` is the start; the UI displays the coverage period. Other columns
use the publication date. The generator sorts newest first, preserves catalog
order for equal dates, and checks article files exist before writing. It updates
the latest three homepage entries and every article in the column archive.
Do not hand-edit these generated lists; add content through the catalog.

Paper navigation is generated from `papers/readings.json`. Each reading has one
primary category; the catalog retains its title, complete archive description,
metadata, article slug and ISO `readDate`. Topics are defined in the same file.

After adding a paper's existing-style `papers/<slug>/index.html` page, prepend
its catalog entry and run:

```bash
python3 scripts/build_papers.py
```

This updates the homepage's latest three readings and topic links, the paper
landing page's latest reading per topic, and the full topic archives at
`papers/topics/<category>/`. Readings sort by reading date descending; entries
with the same date retain catalog order, so prepend newly added readings.
All generated navigation is static HTML and works without JavaScript.
Do not hand-edit generated paper indexes or the homepage paper section.
`papers/archive.css` styles these components without changing article layouts.
Commit the catalog and regenerated HTML together when publishing.

Keep site navigation in English; article bodies can use the language appropriate
to the content. Use relative links and match the existing shared stylesheet.

The fonts are [DM Sans](https://github.com/google/fonts/tree/main/ofl/dmsans)
and [IBM Plex Mono](https://github.com/google/fonts/tree/main/ofl/ibmplexmono), licensed under
the SIL Open Font License; their licenses are included in `assets/`.

## Preview and deployment

Run `python3 -m http.server 8765` in this directory and visit
`http://localhost:8765`.

GitHub Pages publishes the repository root on the `main` branch. Pushing a change
to `main` triggers the existing Pages deployment.
