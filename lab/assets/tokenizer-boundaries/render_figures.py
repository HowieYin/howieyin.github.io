"""Plot measured token IDs and codepoint split census from results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "results.json").read_text())
BLUE, ORANGE, GREEN = "#286aab", "#d17b28", "#35836a"
plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "text.color": "#263346", "svg.hashsalt": "tokenizer-boundaries-20261116"})


def save(fig, name):
    path = ROOT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white",
                metadata={"Date": None, "Description": __doc__})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def token_box(ax, x, y, width, token, raw, color):
    ax.add_patch(FancyBboxPatch((x, y), width, .68, boxstyle="round,pad=.03",
                               facecolor=color, edgecolor="none", alpha=.12))
    ax.text(x + width / 2, y + .44, raw, ha="center", fontsize=14, weight="bold")
    ax.text(x + width / 2, y + .16, str(token), ha="center", fontsize=12, color=color)


def paths():
    rows = DATA["results"][1]["cases"]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set(xlim=(0, 8), ylim=(-.3, 5.5))
    ax.axis("off")
    ax.text(0, 5.2, "Same UTF-8 bytes, different token boundaries", fontsize=15, weight="bold")
    for group, index in enumerate((0, 2)):
        result = rows[index]
        y = 3.5 - group * 2.5
        ax.text(0, y + 1.2, f'Input: {result["text"]}', fontsize=14, weight="bold")
        for offset, key, label, color in [(0, "whole", "Encode whole", BLUE),
                                           (-.9, "separate", "Encode pieces", ORANGE)]:
            ids, raw = result[f"{key}_ids"], result[f"{key}_token_hex"]
            ax.text(0, y + offset + .33, label, fontsize=12, va="center")
            width = 4.5 / len(ids)
            for i, (token, token_hex) in enumerate(zip(ids, raw)):
                token_box(ax, 2.2 + i * (width + .08), y + offset, width, token,
                          bytes.fromhex(token_hex).decode(), color)
    save(fig, "token-paths.svg")


def census():
    labels = ["hello", "[space]hello", "abcde", "antidis...arianism", "Chinese: 4 chars",
              "café (NFC)", "a × 12", "The quick brown fox.", "1234567890"]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    y = np.arange(9)
    for k, r in enumerate(DATA["results"]):
        counts = [x["changed_ids"] for x in r["split_census"]]
        positions = y + (k - .5) * .36
        ax.barh(positions, counts, height=.32, color=[BLUE, ORANGE][k], label=r["encoding"])
        for pos, value, record in zip(positions, counts, r["split_census"]):
            ax.text(value + .35, pos, f'{value}/{record["split_points"]}',
                    va="center", fontsize=11, color=[BLUE, ORANGE][k])
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 28), xlabel="Split points with changed token IDs",
           title="All nonempty codepoint splits in 9 fixed strings")
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e4e9ef")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "split-census.svg")


def prefix():
    result = DATA["results"][1]["cases"][6]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set(xlim=(0, 8.3), ylim=(-.1, 4.3))
    ax.axis("off")
    ax.text(0, 4.05, "Appending 'lo' changes the old tail token", fontsize=15, weight="bold")
    rows = [("Old text: We say hel", result["cached_prefix_ids"]),
            ("Full text: We say hello", result["whole_ids"]),
            ("IDs shared at the start", result["whole_ids"][:2])]
    for row, (label, ids) in enumerate(rows):
        y = 2.7 - row * 1.1
        ax.text(0, y + .9, label, fontsize=12)
        for i, token in enumerate(ids):
            raw = {1687: "We", 2019: "[space]say", 11591: "[space]hel",
                   24748: "[space]hello"}[token]
            token_box(ax, .15 + i * 2.65, y, 2.4, token, raw,
                      GREEN if i < result["reusable_prefix_tokens"] else ORANGE)
    save(fig, "prefix-tail.svg")


if __name__ == "__main__":
    paths()
    census()
    prefix()
