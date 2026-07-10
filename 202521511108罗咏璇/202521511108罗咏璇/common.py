from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml, load_digits
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiment_results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_openml_compatible(name: str, version: int):
    kwargs = {"name": name, "version": version, "as_frame": False}
    if "parser" in inspect.signature(fetch_openml).parameters:
        kwargs["parser"] = "auto"
    return fetch_openml(**kwargs)


def load_high_dimensional_data(
    dataset: str = "digits",
    sample_size: Optional[int] = 1200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load and standardize a high-dimensional dataset.

    The default `digits` dataset is offline and reproducible. MNIST uses
    OpenML and may require network access on the first run.
    """
    dataset = dataset.lower()
    if dataset == "digits":
        data = load_digits()
        x = data.data.astype(np.float64)
        y = data.target.astype(int)
        dataset_label = "sklearn digits"
    elif dataset == "mnist":
        data = fetch_openml_compatible("mnist_784", version=1)
        x = data.data.astype(np.float64)
        y = data.target.astype(int)
        dataset_label = "MNIST"
    elif dataset in {"fashion-mnist", "fashion_mnist"}:
        data = fetch_openml_compatible("Fashion-MNIST", version=1)
        x = data.data.astype(np.float64)
        y = data.target.astype(int)
        dataset_label = "Fashion-MNIST"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if sample_size is not None and sample_size > 0 and sample_size < len(x):
        rng = np.random.default_rng(random_state)
        index = rng.choice(len(x), size=sample_size, replace=False)
        x = x[index]
        y = y[index]

    x = StandardScaler().fit_transform(x)
    return x, y, dataset_label


def evaluate_embedding(
    x: np.ndarray,
    y: np.ndarray,
    embedding: np.ndarray,
    runtime_seconds: float,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute quantitative indicators for a 2D embedding."""
    unique_labels = np.unique(y)
    if len(unique_labels) > 1:
        silhouette = silhouette_score(embedding, y)
    else:
        silhouette = np.nan

    kmeans = KMeans(n_clusters=len(unique_labels), random_state=random_state, n_init=10)
    kmeans.fit(embedding)
    sse = float(kmeans.inertia_)

    trust = trustworthiness(x, embedding, n_neighbors=min(10, len(x) - 1))

    return {
        "silhouette_score": float(silhouette),
        "trustworthiness": float(trust),
        "sse": sse,
        "runtime_seconds": float(runtime_seconds),
    }


def plot_embedding(
    embedding: np.ndarray,
    y: np.ndarray,
    method_name: str,
    dataset_label: str,
    output_path: Path,
) -> None:
    """Save a 2D scatter plot for the embedding."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap="tab10",
        s=12,
        alpha=0.82,
        linewidths=0,
    )
    ax.set_title(f"{method_name} on {dataset_label}", fontsize=13)
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class label", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_reducer(
    method_name: str,
    reducer_factory: Callable[[], object],
    dataset: str = "digits",
    sample_size: Optional[int] = 1200,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run one dimensionality reduction method and save its figure."""
    ensure_output_dirs()
    x, y, dataset_label = load_high_dimensional_data(
        dataset=dataset,
        sample_size=sample_size,
        random_state=random_state,
    )

    reducer = reducer_factory()
    start = time.perf_counter()
    embedding = reducer.fit_transform(x)
    runtime = time.perf_counter() - start

    metrics = evaluate_embedding(x, y, embedding, runtime, random_state=random_state)
    figure_path = FIGURES_DIR / f"{method_name.lower().replace('-', '_')}_result.png"
    plot_embedding(embedding, y, method_name, dataset_label, figure_path)

    result: Dict[str, object] = {
        "method": method_name,
        "dataset": dataset_label,
        "n_samples": int(len(x)),
        "n_features": int(x.shape[1]),
        "figure": str(figure_path.relative_to(PROJECT_ROOT)),
    }
    result.update(metrics)
    return result


def save_results(rows: list[Dict[str, object]]) -> None:
    ensure_output_dirs()
    df = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "metrics.csv"
    summary_path = RESULTS_DIR / "summary.md"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    markdown = [
        "# 实验结果汇总",
        "",
        "表中指标含义：Silhouette Score 越大表示类别分离越明显；Trustworthiness 越大表示低维空间越能保持高维近邻关系；SSE 越小表示 KMeans 聚类内部误差越小；运行时间越短表示效率越高。",
        "",
        dataframe_to_markdown(df),
        "",
    ]
    summary_path.write_text("\n".join(markdown), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default="digits", choices=["digits", "mnist", "fashion-mnist"])
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--min-dist", type=float, default=0.1)
    return parser


def print_result(result: Dict[str, object]) -> None:
    print(f"Method: {result['method']}")
    print(f"Figure: {result['figure']}")
    print(f"Silhouette: {result['silhouette_score']:.4f}")
    print(f"Trustworthiness: {result['trustworthiness']:.4f}")
    print(f"SSE: {result['sse']:.4f}")
    print(f"Runtime seconds: {result['runtime_seconds']:.4f}")
