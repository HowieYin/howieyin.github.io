# Howie Yin

Personal website: **[https://howieyin.github.io/](https://howieyin.github.io/)**

A responsive personal homepage with warm paper tones, editorial typography and
olive accents. About, Projects and Notes are intentionally left open for future
content.

## Editing

- `index.html`: headings and page content. The empty sections have comments showing
  where to add an introduction, projects and notes.
- `styles.css`: layout, colors, responsive styles and motion.
- `assets/`: locally served fonts and favicon. No external services are needed at
  runtime.

The fonts are [Instrument Serif](https://github.com/google/fonts/tree/main/ofl/instrumentserif)
and [DM Sans](https://github.com/google/fonts/tree/main/ofl/dmsans), licensed under
the SIL Open Font License; their licenses are included in `assets/`.

## Preview and deployment

Run `python3 -m http.server 8765` in this directory and visit
`http://localhost:8765`.

GitHub Pages publishes the repository root on the `main` branch. Pushing a change
to `main` triggers the existing Pages deployment.
