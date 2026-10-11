"""Download and prepare the CC BY 4.0 Sentinel-1 oil-spill dataset."""

import argparse
import csv
import hashlib
from io import BytesIO, TextIOWrapper
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np
from PIL import Image

ARCHIVE_URL = "https://zenodo.org/records/22728926/files/oilspill-detection-dataset_20260912-1b9a62c.zip"
ARCHIVE_MD5 = "6ccf0b2c83b3958527e15e34c8c4d5b1"
SPLITS = ("train", "val", "test")


def find_member(archive: zipfile.ZipFile, relative_path: str) -> str:
    suffix = relative_path.replace("\\", "/").strip("/").lower()
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive member ending in {relative_path!r}, found {len(matches)}")
    return matches[0]


def verify_md5(path: Path) -> None:
    digest = hashlib.md5()
    with path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError(f"Dataset archive checksum mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()

    archive_path = args.data_dir / "downloads" / "oilspill-segmentation-dataset.zip"
    output_root = args.data_dir / "processed" / "oilspill-segmentation"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading dataset archive to {archive_path}")
        urlretrieve(ARCHIVE_URL, archive_path)
    verify_md5(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        for split in SPLITS:
            manifest_name = find_member(archive, f"splits/{split}.csv")
            rows = csv.DictReader(TextIOWrapper(archive.open(manifest_name), encoding="utf-8"))
            image_dir = output_root / split / "images"
            mask_dir = output_root / split / "masks"
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for row in rows:
                sample_id = row["sample_id"]
                image_member = find_member(archive, row["image_path"])
                mask_member = find_member(archive, row["mask_path"])
                with Image.open(BytesIO(archive.read(image_member))) as source_image:
                    source_image.convert("L").save(image_dir / f"{sample_id}.png")
                with Image.open(BytesIO(archive.read(mask_member))) as source_mask:
                    binary_mask = (np.asarray(source_mask.convert("L"), dtype=np.uint8) > 0).astype(np.uint8) * 255
                    Image.fromarray(binary_mask, mode="L").save(mask_dir / f"{sample_id}.png")
                count += 1
            print(f"Prepared {split}: {count} paired tiles")

    (output_root / "DATASET_SOURCE.txt").write_text(
        "Oil Spill Segmentation Dataset (800x600), Sentinel-1, Cantarell, 2020.\n"
        "Source: https://doi.org/10.5281/zenodo.22214144\n"
        "Author: Garcia, Jose Roberto Motta.\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/\n"
        "Changes: source RGB tiles converted to grayscale; binary labels normalized to 0/255.\n",
        encoding="utf-8",
    )
    print(f"Prepared dataset at {output_root}")


if __name__ == "__main__":
    main()