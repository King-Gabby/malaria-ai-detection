"""
download_dataset.py — Download the BBBC041 malaria bounding-box dataset.

The Broad Bioimage Benchmark Collection (BBBC041) provides:
  • ~1,364 blood smear images (.png)
  • JSON annotation file with bounding boxes for parasites and RBCs.

This script automates the download + extraction so the rest of the pipeline
has a deterministic input path (data/raw/).

Usage:
    python -m src.data.download_dataset --output_dir data/raw
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# BBBC041 URLs (Broad Institute public S3 bucket)
# The dataset is split across images + a single JSON annotation file.
# ---------------------------------------------------------------------------
BBBC041_IMAGES_URL = (
    "https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip"
)


def download_file(url: str, dest_path: Path) -> None:
    """Stream-download a large file with a progress bar.

    We stream in 8 KB chunks to avoid loading the entire ZIP into memory,
    which matters on machines with limited RAM.
    """
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP archive, then clean up the archive file."""
    print(f"Extracting {zip_path.name} → {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    zip_path.unlink()  # Remove ZIP after extraction to save disk space
    print("Extraction complete.")


def download_bbbc041(output_dir: str = "data/raw") -> Path:
    """End-to-end download of the BBBC041 dataset.

    Returns:
        Path to the directory containing raw images and annotations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zip_dest = output_path / "malaria.zip"

    # Skip if images already exist (idempotent re-runs)
    if any(output_path.rglob("*.png")):
        print(f"Images already present in {output_path}. Skipping download.")
        return output_path

    print(f"Downloading BBBC041 dataset → {output_path} ...")
    download_file(BBBC041_IMAGES_URL, zip_dest)
    extract_zip(zip_dest, output_path)

    print(f"Dataset ready at {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the BBBC041 malaria bounding-box dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save the raw dataset (default: data/raw).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_bbbc041(args.output_dir)
