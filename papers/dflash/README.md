# DFlash close reading

Published route: https://howieyin.github.io/papers/dflash/

`index.md` is the Chinese source. `index.html` is the standalone static output,
using the existing site's `../../styles.css` and local font assets.
No client-side CDN, framework build or JavaScript math renderer is required.

All5originalfigures and13tables are preserved. Provenance / checksums:
`sources/asset-manifest.json`; originalPDF andTeX retained under`sources/`.
Generated diagrams have Mermaid source in`assets/generated/`.

Rebuild with Python dependencies `beautifulsoup4`, `markdown`, `latex2mathml`,
and `pygments`:

```sh
python sources/render-reading.py
```

The two Mermaid SVGs are already rendered; rebuild does not access the network.
To preview the integrated repository, run its existing static-server command:

```sh
python3 -m http.server 8776
```

Calculation checks:

```sh
node interactive/model-check.cjs
```

Review notes distinguish paper measurements, external sources, analytical
examples and implementation checks. Real-browser and deployment evidence
is retained in the workspace publication report.
