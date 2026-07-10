from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from assignment_config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_directories


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    return plt, sns


def save_distribution_figure(plt, sns) -> None:
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["activity"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["activity"]
    frame = pd.DataFrame(
        {
            "activity": pd.concat([y_train, y_test], ignore_index=True),
            "split": ["train"] * len(y_train) + ["test"] * len(y_test),
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=frame, y="activity", hue="split", ax=ax, palette=["#3B82F6", "#F97316"])
    ax.set_title("Class distribution of UCI HAR dataset")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Activity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=220)
    plt.close(fig)


def save_embedding_figures(plt, sns) -> None:
    for path in sorted(RESULTS_DIR.glob("embedding_*_2d.csv")):
        frame = pd.read_csv(path)
        method_label = path.stem.replace("embedding_", "").replace("_2d", "").replace("_", " ").upper()
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        sns.scatterplot(
            data=frame,
            x="x",
            y="y",
            hue="activity",
            style="split",
            s=18,
            linewidth=0,
            alpha=0.78,
            ax=ax,
        )
        ax.set_title(f"{method_label} 2D embedding")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(loc="best", fontsize=7, frameon=True)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{path.stem}.png", dpi=220)
        plt.close(fig)


def save_metric_barplots(plt, sns) -> None:
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    valid = metrics[metrics["evaluation_mode"] != "skipped"].copy()
    valid["method_dim"] = valid["method"] + " " + valid["dimensions"].astype(str) + "D"

    preferred = valid[(valid["dimensions"].isin([30, 561])) | (valid["method"].isin(["MDS", "t-SNE"]))]
    preferred = preferred.sort_values(["accuracy", "macro_f1"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=preferred, y="method_dim", x="accuracy", ax=ax, color="#2563EB")
    ax.set_xlim(0, 1)
    ax.set_title("KNN accuracy after dimensionality reduction")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_comparison.png", dpi=220)
    plt.close(fig)

    tw = valid[valid["dimensions"] == 2].dropna(subset=["trustworthiness"]).sort_values("trustworthiness", ascending=False)
    if not tw.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        sns.barplot(data=tw, y="method", x="trustworthiness", ax=ax, color="#059669")
        ax.set_xlim(0, 1)
        ax.set_title("Neighborhood trustworthiness of 2D embeddings")
        ax.set_xlabel("Trustworthiness")
        ax.set_ylabel("")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "trustworthiness_comparison.png", dpi=220)
        plt.close(fig)


def save_confusion_figures(plt, sns) -> None:
    preferred = [RESULTS_DIR / "confusion_raw_561d_561d.csv"]
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    valid = metrics[(metrics["evaluation_mode"] != "skipped") & (metrics["method"] != "Raw 561D")].dropna(subset=["accuracy"])
    if not valid.empty:
        best = valid.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]
        preferred.append(RESULTS_DIR / f"confusion_{best['slug']}_{int(best['dimensions'])}d.csv")

    for path in preferred:
        if not path.exists():
            continue
        matrix = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(path.stem.replace("confusion_", "Confusion matrix: ").replace("_", " "))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{path.stem}.png", dpi=220)
        plt.close(fig)


def main() -> int:
    ensure_directories()
    if not (RESULTS_DIR / "metrics.csv").exists():
        raise FileNotFoundError("Run code/run_experiments.py first.")
    plt, sns = configure_matplotlib()
    save_distribution_figure(plt, sns)
    save_embedding_figures(plt, sns)
    save_metric_barplots(plt, sns)
    save_confusion_figures(plt, sns)
    print(f"Figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
