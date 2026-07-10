from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from assignment_config import ACTIVITY_LABELS, DATASET_FOLDER, PROCESSED_DIR, ensure_directories


def make_unique_feature_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        unique.append(name if counts[name] == 1 else f"{name}__{counts[name]}")
    return unique


def standardize_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    means = train.mean(axis=0)
    stds = train.std(axis=0, ddof=0)
    constant_columns = list(stds[stds == 0].index)
    safe_stds = stds.mask(stds == 0, 1.0)

    train_scaled = (train - means) / safe_stds
    test_scaled = (test - means) / safe_stds

    stats = {
        "constant_columns": constant_columns,
        "mean_preview": means.head(5).to_dict(),
        "std_preview": stds.head(5).to_dict(),
    }
    return train_scaled, test_scaled, stats


def build_metadata_summary(
    train_rows: int,
    test_rows: int,
    feature_count: int,
    label_count: int,
    constant_columns: list[str],
) -> dict[str, object]:
    return {
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "total_rows": int(train_rows + test_rows),
        "feature_count": int(feature_count),
        "label_count": int(label_count),
        "constant_columns": constant_columns,
    }


def read_features(path: Path) -> list[str]:
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["idx", "feature"])
    return make_unique_feature_names(raw["feature"].astype(str).tolist())


def read_matrix(path: Path, feature_names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=feature_names)


def read_labels(path: Path) -> pd.Series:
    labels = pd.read_csv(path, header=None).iloc[:, 0].astype(int)
    return labels.map(ACTIVITY_LABELS)


def assert_dataset_files() -> None:
    required = [
        DATASET_FOLDER / "features.txt",
        DATASET_FOLDER / "activity_labels.txt",
        DATASET_FOLDER / "train" / "X_train.txt",
        DATASET_FOLDER / "train" / "y_train.txt",
        DATASET_FOLDER / "test" / "X_test.txt",
        DATASET_FOLDER / "test" / "y_test.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Run code/download_data.py first. Missing files:\n{joined}")


def main() -> int:
    ensure_directories()
    assert_dataset_files()

    feature_names = read_features(DATASET_FOLDER / "features.txt")
    train = read_matrix(DATASET_FOLDER / "train" / "X_train.txt", feature_names)
    test = read_matrix(DATASET_FOLDER / "test" / "X_test.txt", feature_names)
    y_train = read_labels(DATASET_FOLDER / "train" / "y_train.txt")
    y_test = read_labels(DATASET_FOLDER / "test" / "y_test.txt")

    train_scaled, test_scaled, stats = standardize_train_test(train, test)
    metadata = build_metadata_summary(
        train_rows=len(train_scaled),
        test_rows=len(test_scaled),
        feature_count=train_scaled.shape[1],
        label_count=len(ACTIVITY_LABELS),
        constant_columns=stats["constant_columns"],
    )
    metadata["activity_labels"] = list(ACTIVITY_LABELS.values())
    metadata["standardization"] = {
        "fit_on": "official train split",
        "constant_column_count": len(stats["constant_columns"]),
    }

    train_scaled.to_csv(PROCESSED_DIR / "X_train_scaled.csv.gz", index=False, compression="gzip")
    test_scaled.to_csv(PROCESSED_DIR / "X_test_scaled.csv.gz", index=False, compression="gzip")
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False, header=["activity"])
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False, header=["activity"])
    pd.DataFrame({"feature": feature_names}).to_csv(PROCESSED_DIR / "features.csv", index=False)
    pd.DataFrame(
        {"label_id": list(ACTIVITY_LABELS.keys()), "activity": list(ACTIVITY_LABELS.values())}
    ).to_csv(PROCESSED_DIR / "activity_labels.csv", index=False)
    (PROCESSED_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
