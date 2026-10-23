# Howie Yin

Personal website: **[https://howieyin.github.io/](https://howieyin.github.io/)**

A minimal personal homepage with a dark background, sans-serif typography,
monospace details and muted blue links. About, Projects and Notes are
intentionally left empty for future content.

## Editing

- `index.html`: headings and page content. The empty sections have comments showing
  where to add an introduction, projects and notes.
- `styles.css`: layout, colors and responsive styles.
- `assets/`: locally served fonts and favicon. No external services are needed at
  runtime.

The fonts are [DM Sans](https://github.com/google/fonts/tree/main/ofl/dmsans)
and [IBM Plex Mono](https://github.com/google/fonts/tree/main/ofl/ibmplexmono), licensed under
the SIL Open Font License; their licenses are included in `assets/`.

## Preview and deployment

Run `python3 -m http.server 8765` in this directory and visit
`http://localhost:8765`.

GitHub Pages publishes the repository root on the `main` branch. Pushing a change
to `main` triggers the existing Pages deployment.
