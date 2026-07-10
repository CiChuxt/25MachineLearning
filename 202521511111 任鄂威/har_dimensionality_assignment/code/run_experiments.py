from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from assignment_config import DIMENSIONALITY_METHODS, PROCESSED_DIR, RESULTS_DIR, ensure_directories


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required = [
        PROCESSED_DIR / "X_train_scaled.csv.gz",
        PROCESSED_DIR / "X_test_scaled.csv.gz",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_test.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run code/prepare_data.py first. Missing:\n" + "\n".join(missing))
    x_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv.gz")
    x_test = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv.gz")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["activity"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["activity"]
    return x_train, x_test, y_train, y_test


def stratified_sample(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    x_all = pd.concat([x_train, x_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    rng = np.random.default_rng(seed)

    selected: list[int] = []
    per_class = max(1, sample_size // y_all.nunique())
    for label in sorted(y_all.unique()):
        indices = np.flatnonzero((y_all == label).to_numpy())
        take = min(per_class, len(indices))
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(selected) < sample_size:
        remaining = np.setdiff1d(np.arange(len(y_all)), np.array(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=min(sample_size - len(selected), len(remaining)), replace=False).tolist())

    selected = sorted(selected[:sample_size])
    x_sample = x_all.iloc[selected].to_numpy(dtype=np.float64)
    y_sample = y_all.iloc[selected].to_numpy()
    indices = np.arange(len(selected))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=seed, stratify=y_sample)
    return x_sample, y_sample, train_idx, test_idx, x_sample[train_idx], x_sample[test_idx]


def make_estimator(method: str, dimensions: int, seed: int):
    from sklearn.decomposition import KernelPCA, PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE

    if method == "PCA":
        return PCA(n_components=dimensions, random_state=seed)
    if method == "Kernel PCA":
        return KernelPCA(n_components=dimensions, kernel="rbf", gamma=0.01, eigen_solver="arpack", random_state=seed)
    if method == "Isomap":
        return Isomap(n_components=dimensions, n_neighbors=12)
    if method == "LLE":
        return LocallyLinearEmbedding(
            n_components=dimensions,
            n_neighbors=12,
            method="standard",
            eigen_solver="arpack",
            random_state=seed,
        )
    if method == "MDS":
        return MDS(n_components=dimensions, random_state=seed, n_init=1, max_iter=120, normalized_stress="auto")
    if method == "t-SNE":
        return TSNE(
            n_components=dimensions,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=30,
            max_iter=900,
        )
    if method == "UMAP":
        try:
            from umap import UMAP
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(f"UMAP is unavailable: {exc}") from exc
        return UMAP(n_components=dimensions, n_neighbors=15, min_dist=0.1, random_state=seed)
    raise ValueError(f"Unknown method: {method}")


def fit_embedding(method: str, dimensions: int, supports_transform: bool, seed: int, x_train, x_test, x_all):
    estimator = make_estimator(method, dimensions, seed)
    if supports_transform and hasattr(estimator, "transform"):
        train_embedding = estimator.fit_transform(x_train)
        test_embedding = estimator.transform(x_test)
        all_embedding = np.vstack([train_embedding, test_embedding])
        mode = "inductive_train_transform"
    elif supports_transform:
        all_embedding = estimator.fit_transform(x_all)
        train_embedding = all_embedding[: len(x_train)]
        test_embedding = all_embedding[len(x_train) :]
        mode = "fit_transform_combined_no_transform_method"
    else:
        all_embedding = estimator.fit_transform(x_all)
        train_embedding = all_embedding[: len(x_train)]
        test_embedding = all_embedding[len(x_train) :]
        mode = "transductive_sample_embedding"
    return train_embedding, test_embedding, all_embedding, mode


def evaluate_classifier(x_train, x_test, y_train, y_test):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "predictions": predictions,
        "confusion": confusion_matrix(y_test, predictions, labels=sorted(np.unique(y_train))),
        "labels": sorted(np.unique(y_train)),
    }


def compute_optional_embedding_scores(x_high, y, embedding):
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score

    scores = {"trustworthiness": np.nan, "silhouette": np.nan}
    if embedding.shape[1] >= 2 and len(np.unique(y)) > 1:
        try:
            scores["trustworthiness"] = float(trustworthiness(x_high, embedding, n_neighbors=10))
        except Exception:
            scores["trustworthiness"] = np.nan
        try:
            scores["silhouette"] = float(silhouette_score(embedding[:, :2], y))
        except Exception:
            scores["silhouette"] = np.nan
    return scores


def save_embedding(slug: str, dimensions: int, embedding, labels, train_size: int) -> None:
    if dimensions != 2:
        return
    split = np.array(["train"] * train_size + ["test"] * (len(labels) - train_size))
    frame = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "activity": labels,
            "split": split,
        }
    )
    frame.to_csv(RESULTS_DIR / f"embedding_{slug}_2d.csv", index=False)


def save_confusion(slug: str, dimensions: int, labels: list[str], matrix: np.ndarray) -> None:
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.to_csv(RESULTS_DIR / f"confusion_{slug}_{dimensions}d.csv", encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=1500)
    args = parser.parse_args()

    ensure_directories()
    x_train_full, x_test_full, y_train_full, y_test_full = load_processed_data()
    x_sample, y_sample, train_idx, test_idx, x_train, x_test = stratified_sample(
        x_train_full, x_test_full, y_train_full, y_test_full, args.sample_size, args.seed
    )
    y_train = y_sample[train_idx]
    y_test = y_sample[test_idx]
    x_all_for_split = np.vstack([x_train, x_test])
    y_all_for_split = np.concatenate([y_train, y_test])

    metrics: list[dict[str, object]] = []
    warnings: list[str] = []

    start = time.perf_counter()
    raw_eval = evaluate_classifier(x_train, x_test, y_train, y_test)
    raw_runtime = time.perf_counter() - start
    save_confusion("raw_561d", 561, raw_eval["labels"], raw_eval["confusion"])
    metrics.append(
        {
            "method": "Raw 561D",
            "slug": "raw_561d",
            "dimensions": 561,
            "evaluation_mode": "sample_train_test_split",
            "accuracy": raw_eval["accuracy"],
            "macro_f1": raw_eval["macro_f1"],
            "trustworthiness": np.nan,
            "silhouette": np.nan,
            "runtime_seconds": raw_runtime,
            "notes": "KNN baseline on standardized original features.",
        }
    )

    for spec in DIMENSIONALITY_METHODS:
        dimensions_to_run = sorted(set(spec.visualization_dimensions + spec.classification_dimensions))
        for dimensions in dimensions_to_run:
            start = time.perf_counter()
            try:
                train_embedding, test_embedding, all_embedding, mode = fit_embedding(
                    spec.name,
                    dimensions,
                    spec.supports_transform,
                    args.seed,
                    x_train,
                    x_test,
                    x_all_for_split,
                )
                classifier_eval = evaluate_classifier(train_embedding, test_embedding, y_train, y_test)
                runtime = time.perf_counter() - start
                scores = compute_optional_embedding_scores(x_all_for_split, y_all_for_split, all_embedding)
                save_embedding(spec.slug, dimensions, all_embedding, y_all_for_split, len(y_train))
                save_confusion(spec.slug, dimensions, classifier_eval["labels"], classifier_eval["confusion"])
                metrics.append(
                    {
                        "method": spec.name,
                        "slug": spec.slug,
                        "dimensions": dimensions,
                        "evaluation_mode": mode,
                        "accuracy": classifier_eval["accuracy"],
                        "macro_f1": classifier_eval["macro_f1"],
                        "trustworthiness": scores["trustworthiness"],
                        "silhouette": scores["silhouette"],
                        "runtime_seconds": runtime,
                        "notes": "2D rows are used for visualization; higher dimensions are used for classification when supported.",
                    }
                )
                print(f"{spec.name} {dimensions}D complete in {runtime:.1f}s")
            except Exception as exc:
                message = f"{spec.name} {dimensions}D skipped: {exc}"
                warnings.append(message)
                metrics.append(
                    {
                        "method": spec.name,
                        "slug": spec.slug,
                        "dimensions": dimensions,
                        "evaluation_mode": "skipped",
                        "accuracy": np.nan,
                        "macro_f1": np.nan,
                        "trustworthiness": np.nan,
                        "silhouette": np.nan,
                        "runtime_seconds": time.perf_counter() - start,
                        "notes": message,
                    }
                )
                print(message)

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    run_info = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "train_sample_rows": int(len(y_train)),
        "test_sample_rows": int(len(y_test)),
        "methods": [asdict(method) for method in DIMENSIONALITY_METHODS],
        "warnings": warnings,
    }
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
