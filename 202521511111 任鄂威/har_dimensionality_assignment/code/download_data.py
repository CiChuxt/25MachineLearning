from __future__ import annotations

import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from assignment_config import DATASET_FOLDER, DATASET_PAGE, DATASET_URLS, DATASET_ZIP, RAW_DIR, ensure_directories


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def extract_nested_archives(root: Path) -> None:
    nested = list(root.rglob("*.zip"))
    for archive_path in nested:
        if archive_path.resolve() == DATASET_ZIP.resolve():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(archive_path.parent)


def dataset_is_present() -> bool:
    expected = [
        DATASET_FOLDER / "features.txt",
        DATASET_FOLDER / "activity_labels.txt",
        DATASET_FOLDER / "train" / "X_train.txt",
        DATASET_FOLDER / "test" / "X_test.txt",
    ]
    return all(path.exists() for path in expected)


def write_source_notes(url: str) -> None:
    notes = {
        "dataset": "UCI Human Activity Recognition Using Smartphones Dataset",
        "dataset_page": DATASET_PAGE,
        "download_url": url,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "license_note": "Dataset redistributed for course analysis with source attribution.",
    }
    (RAW_DIR / "source_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ensure_directories()
    if dataset_is_present():
        print(f"Dataset already present: {DATASET_FOLDER}")
        return 0

    errors: list[str] = []
    for url in DATASET_URLS:
        try:
            print(f"Downloading: {url}")
            download_file(url, DATASET_ZIP)
            extract_zip(DATASET_ZIP, RAW_DIR)
            extract_nested_archives(RAW_DIR)
            if dataset_is_present():
                write_source_notes(url)
                print(f"Dataset ready: {DATASET_FOLDER}")
                return 0
            errors.append(f"{url}: archive downloaded but expected UCI HAR files were not found")
        except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile, OSError) as exc:
            errors.append(f"{url}: {exc}")

    print("Unable to download or extract the UCI HAR dataset.", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
