"""Rebuild native SVG figures; schedule counts are not hardware timings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "results.json").read_text())
BLUE, ORANGE, GREEN = "#286aab", "#d17b28", "#35836a"
plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelcolor": "#263346",
                     "text.color": "#263346", "axes.titleweight": "bold",
                     "svg.hashsalt": "chunked-prefill-20261116"})


def save(fig, name):
    path = ROOT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white",
                metadata={"Date": None, "Description": __doc__})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def schedule():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for row, result in enumerate(DATA["budget_traces"]):
        y = 2 - row
        for step, record in enumerate(result["trace"]):
            left = step + .12
            unit = .048
            for i in range(record["total_tokens"]):
                ax.add_patch(Rectangle((left + i * unit, y - .1), unit * .92, .34,
                                       facecolor=BLUE if i < 2 else ORANGE))
            ax.text(step + .46, y - .33,
                    f'D2 + P{record["prefill_tokens"]}', ha="center", fontsize=11)
    ax.set(xlim=(0, 4), ylim=(-.65, 2.65),
           xticks=np.arange(4) + .46, xticklabels=["1", "2", "3", "4"],
           yticks=[2, 1, 0], yticklabels=["Budget 5", "Budget 8", "Budget 16"],
           xlabel="Scheduling iteration (not elapsed time)",
           title="12 prompt tokens + 2 active decode streams")
    ax.tick_params(axis="y", length=0, pad=14)
    ax.set_title("12 prompt tokens + 2 active decode streams", y=1.16)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.legend(handles=[Patch(color=BLUE, label="Decode query"),
                       Patch(color=ORANGE, label="Prefill query")],
              loc="upper center", bbox_to_anchor=(.5, 1.06), ncol=2,
              frameon=False, fontsize=11)
    save(fig, "schedule.svg")


def causal():
    mask = np.zeros((12, 12), dtype=int)
    for i in range(12):
        for j in range(i + 1):
            mask[i, j] = 1 if j < (i // 4) * 4 else 2
    fig, ax = plt.subplots(figsize=(6.8, 6.1))
    ax.imshow(mask, cmap=ListedColormap(["#f5f6f8", ORANGE, BLUE]), vmin=0, vmax=2)
    ax.set(xticks=range(12), xticklabels=range(1, 13), yticks=range(12),
           yticklabels=range(1, 13), xlabel="Absolute key position",
           ylabel="Absolute query position", title="Chunk size 4: keep all earlier KV")
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    for edge in (3.5, 7.5):
        ax.axhline(edge, color="#263346", lw=1.4)
        ax.axvline(edge, color="#263346", lw=1.4, alpha=.45)
    ax.legend(handles=[Patch(color=ORANGE, label="Earlier-chunk KV"),
                       Patch(color=BLUE, label="Current-chunk KV"),
                       Patch(facecolor="#f5f6f8", edgecolor="#bfc8d2", label="Masked future")],
              loc="upper center", bbox_to_anchor=(.5, -.15), ncol=1,
              frameon=False, fontsize=11)
    save(fig, "causal-kv.svg")


def ablation():
    # Exact transcription of Sarathi-Serve v3 Table 4, not a new benchmark.
    labels = ["Chat / hybrid", "Chat / chunks", "Chat / combined",
              "ArXiv / hybrid", "ArXiv / chunks", "ArXiv / combined"]
    values = [[.53, 1.04, .76, 3.78, 5.38, 3.90],
              [.68, .17, .14, 1.38, .20, .17]]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7), sharey=True)
    for k, (ax, numbers) in enumerate(zip(axes, values)):
        ax.barh(np.arange(6), numbers, height=.6, color=[BLUE, ORANGE, GREEN] * 2)
        ax.set(yticks=np.arange(6), yticklabels=labels, xlabel="Seconds (lower is better)",
               title=["P50 first-token latency", "P99 inter-token latency"][k])
        ax.set_xlim(0, [6.2, 1.63][k])
        for i, v in enumerate(numbers):
            ax.text(v + [.09, .024][k], i, f"{v:.2f}", va="center", fontsize=11)
        ax.axhline(2.5, color="#bac4ce", lw=.7)
        ax.grid(axis="x", color="#e5e9ef", zorder=0)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Sarathi-Serve Table 4: Yi-34B / 2 A100 / budget 1024",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    save(fig, "paper-ablation.svg")


if __name__ == "__main__":
    schedule()
    causal()
    ablation()
