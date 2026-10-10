"""Matplotlib figures for the notebooks (consistent style, computed from published artifacts only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {"figure.figsize": (11, 4.2), "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
         "font.size": 10, "axes.titleweight": "bold"}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def show_png(path: Path, title: str | None = None, width: float = 11) -> None:
    """Display a saved figure (charts only - dataset images are never committed)."""
    img = plt.imread(str(path))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(width, width * h / w))
    ax.imshow(img); ax.axis("off")
    if title:
        ax.set_title(title)
    plt.show()


def training_losses(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, (k, t) in enumerate((("box_loss", "Box regression loss"), ("cls_loss", "Classification loss"), ("dfl_loss", "Distribution focal loss"))):
        ax[i].plot(df["epoch"], df[f"train/{k}"], label="train")
        ax[i].plot(df["epoch"], df[f"val/{k}"], label="validation")
        ax[i].axvline(190.5, color="grey", ls=":", lw=1, label="mosaic closed")
        ax[i].set_title(t); ax[i].set_xlabel("epoch"); ax[i].legend()
    fig.tight_layout(); plt.show()


def validation_metrics(df: pd.DataFrame, best_epoch: int = 111) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(df["epoch"], df["metrics/precision(B)"], label="precision"); ax[0].plot(df["epoch"], df["metrics/recall(B)"], label="recall")
    ax[0].set_title("Ultralytics P / R (no ignore regions)")
    ax[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50"); ax[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    ax[1].axvline(best_epoch, color="tab:red", ls="--", lw=1, label=f"selected epoch {best_epoch}")
    ax[1].set_title("Ultralytics mAP (PROVISIONAL)")
    ax[2].plot(df["epoch"], df["lr/pg0"]); ax[2].set_title("Learning rate (SGD, cosine)")
    for a in ax:
        a.set_xlabel("epoch"); a.legend() if a.get_legend_handles_labels()[0] else None
    fig.tight_layout(); plt.show()


def checkpoint_curve(cmp: pd.DataFrame) -> None:
    s = cmp[cmp["checkpoint"].str.startswith("epoch_completed_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    for col, lab in (("AP50 primary", "AP50 - primary (REVIEW ignored)"), ("AP50 secondary", "AP50 - secondary (+ PROPOSED sub-threshold)"),
                     ("AP50 no-ignores", "AP50 - no ignore regions")):
        ax.plot(s["epoch"], s[col], "o-", ms=3, label=lab)
    b = cmp[cmp["checkpoint"] == "best.pt"].iloc[0]
    ax.plot([111], [b["AP50 primary"]], "r*", ms=14, label=f"best.pt (epoch 111): {b['AP50 primary']:.3f}")
    ax.set_xlabel("completed epochs"); ax.set_ylabel("AP50 (val)"); ax.set_title("22-checkpoint ignore-aware comparison (validation only)")
    ax.legend(fontsize=8); fig.tight_layout(); plt.show()


def threshold_tradeoff(tab: pd.DataFrame, frozen_conf: float = 0.20) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(tab["conf"], tab["precision"], "o-", label="precision"); ax[0].plot(tab["conf"], tab["recall"], "o-", label="recall")
    ax[0].plot(tab["conf"], tab["f1"], "o-", label="F1"); ax[0].plot(tab["conf"], tab["f2"], "o--", label="F2")
    ax[0].axvline(frozen_conf, color="tab:red", ls="--", lw=1, label=f"frozen conf {frozen_conf}")
    ax[0].set_xlabel("confidence threshold"); ax[0].set_title("Operating point (val, primary policy)"); ax[0].legend(fontsize=8)
    ax[1].bar(tab["conf"].astype(str), tab["fp_per_image"], color=["tab:red" if c == frozen_conf else "tab:grey" for c in tab["conf"]])
    ax[1].set_xlabel("confidence threshold"); ax[1].set_title("False positives per image (val)")
    fig.tight_layout(); plt.show()


def size_recall(tab: pd.DataFrame) -> None:
    x = np.arange(len(tab)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w / 2, tab["recall val"], w, label="validation (selection)")
    ax.bar(x + w / 2, tab["recall test"], w, label="test (once-only)")
    for i, (n, r) in enumerate(zip(tab["GT (test)"], tab["recall test"])):
        ax.text(i + w / 2, r + 0.01, f"n={n:,}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(tab["size bin (px^2)"]); ax.set_ylim(0, 1); ax.set_ylabel("recall @ conf 0.20")
    ax.set_title("Recall by rock size (primary policy)"); ax.legend(); fig.tight_layout(); plt.show()


def draw_detections(image: np.ndarray, boxes: np.ndarray, title: str = "") -> None:
    """Overlay (N,5) [x0,y0,x1,y1,conf] boxes on an image (local use: dataset pixels are not committed)."""
    fig, ax = plt.subplots(figsize=(11, 11 * image.shape[0] / image.shape[1]))
    ax.imshow(image); ax.axis("off")
    for x0, y0, x1, y1, c in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="magenta", lw=1))
        ax.text(x0, y0 - 2, f"{c:.2f}", color="magenta", fontsize=6)
    ax.set_title(title); plt.show()
