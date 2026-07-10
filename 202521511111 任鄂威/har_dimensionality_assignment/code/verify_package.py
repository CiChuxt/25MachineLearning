from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

from assignment_config import FIGURES_DIR, PROCESSED_DIR, REFERENCES, REPORT_FILE, RESULTS_DIR, ensure_directories


def require_file(path: Path, errors: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"Missing or empty file: {path}")


def verify_docx_text(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing report: {path}")
        return
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    required_text = [
        "基于UCI HAR智能手机人体活动识别数据的高维特征降维方法对比研究",
        "姓名：待补充",
        "学号：待补充",
        "摘要：",
        "关键词：",
        "研究背景",
        "研究方法",
        "数据描述",
        "结果对比与分析",
        "参考文献",
    ]
    for text in required_text:
        if text not in xml:
            errors.append(f"Report text not found: {text}")


def main() -> int:
    ensure_directories()
    errors: list[str] = []
    for path in [
        PROCESSED_DIR / "metadata.json",
        PROCESSED_DIR / "X_train_scaled.csv.gz",
        PROCESSED_DIR / "X_test_scaled.csv.gz",
        RESULTS_DIR / "metrics.csv",
        RESULTS_DIR / "run_info.json",
        FIGURES_DIR / "accuracy_comparison.png",
        FIGURES_DIR / "class_distribution.png",
        REPORT_FILE,
    ]:
        require_file(path, errors)

    if len(REFERENCES) < 15:
        errors.append(f"Reference count is {len(REFERENCES)}, expected at least 15.")

    metrics_path = RESULTS_DIR / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        required_columns = {
            "method",
            "dimensions",
            "evaluation_mode",
            "accuracy",
            "macro_f1",
            "trustworthiness",
            "silhouette",
            "runtime_seconds",
            "notes",
        }
        missing = required_columns.difference(metrics.columns)
        if missing:
            errors.append(f"metrics.csv missing columns: {sorted(missing)}")
        if metrics.empty:
            errors.append("metrics.csv is empty.")
        if "Raw 561D" not in set(metrics["method"]):
            errors.append("Raw 561D baseline is missing.")
        valid_methods = set(metrics.loc[metrics["evaluation_mode"] != "skipped", "method"])
        if len(valid_methods.difference({"Raw 561D"})) < 3:
            errors.append("Fewer than three dimensionality reduction methods produced valid results.")

    verify_docx_text(REPORT_FILE, errors)

    if errors:
        print("Package verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    summary = {
        "report": str(REPORT_FILE),
        "metrics": str(metrics_path),
        "figure_count": len(list(FIGURES_DIR.glob("*.png"))),
        "reference_count": len(REFERENCES),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
