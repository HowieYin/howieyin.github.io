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
  the evolution of ideas and research surveys.
- **AI Lab** (`lab/`): practical tricks, reproducible experiments,
  counterintuitive findings and myth testing.
- **Paper Readings** (`papers/`): a personal archive of papers read, with original
  sources, reading dates, methods, experiments, limitations and personal insights.
  Do not automatically count news coverage or knowledge articles as completed
  paper readings. Each reading should have its own page and archive entry.

Each column has a homepage section and a standalone archive page. The archives
are explicitly empty until real articles are published; topic descriptions are
editorial directions, not published content.

## Editing

- `index.html`: English homepage, column cards, About and Projects.
- `weekly/index.html`, `foundations/index.html`, `lab/index.html`, `papers/index.html`: column landing
  pages. Replace the empty archive block with links to published articles when
  content is ready, and update the homepage publication status accordingly.
- `styles.css`: layout, colors and responsive styles.
- `assets/`: locally served fonts and favicon. No external services are needed at
runtime.

For a new article, add its static HTML file inside the relevant column directory,
then link to it from that column's archive. Keep site navigation in English;
article bodies can use the language appropriate to the content. Use relative
links and match the existing shared stylesheet.

The fonts are [DM Sans](https://github.com/google/fonts/tree/main/ofl/dmsans)
and [IBM Plex Mono](https://github.com/google/fonts/tree/main/ofl/ibmplexmono), licensed under
the SIL Open Font License; their licenses are included in `assets/`.

## Preview and deployment

Run `python3 -m http.server 8765` in this directory and visit
`http://localhost:8765`.

GitHub Pages publishes the repository root on the `main` branch. Pushing a change
to `main` triggers the existing Pages deployment.
