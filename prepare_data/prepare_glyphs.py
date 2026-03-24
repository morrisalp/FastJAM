"""
Prepare a set of glyph images from the pnp dataset for FastJAM.

Reads .tif/.jpg glyph images for a given letter, converts them to RGB PNGs,
and generates Otsu-threshold masks (white=glyph, black=background).

Output structure mirrors the SPair format:
  data/glyph_sets/glyph_{LETTER}/test/
    images/img_000.png ...
    masks/img_000_mask.png ...

Usage:
  uv run prepare_data/prepare_glyphs.py --letter T
  uv run prepare_data/prepare_glyphs.py --letter T --n 27 --source_dir ../pnp/data/leviathan_matching_test_set_preprocessed
"""

import argparse
import os
import sys
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image


def otsu_mask(img_gray: np.ndarray) -> np.ndarray:
    """
    Compute an Otsu mask for a grayscale image.
    Returns a uint8 mask: 255 where the glyph is, 0 background.
    Assumes glyph is darker than background (typical for scanned type).
    """
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def prepare_glyphs(letter: str, source_dir: str, output_base: str, n: int, size: int, seed: int):
    source_path = Path(source_dir) / letter / "test"
    if not source_path.exists():
        # Try uppercase and lowercase
        for candidate in [letter.upper(), letter.lower()]:
            p = Path(source_dir) / candidate / "test"
            if p.exists():
                source_path = p
                break
        else:
            print(f"Error: could not find {source_path}")
            sys.exit(1)

    all_files = sorted([
        f for f in source_path.iterdir()
        if f.suffix.lower() in (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    ])

    if len(all_files) == 0:
        print(f"No image files found in {source_path}")
        sys.exit(1)

    if n is not None and n < len(all_files):
        rng = random.Random(seed)
        all_files = sorted(rng.sample(all_files, n))

    print(f"Using {len(all_files)} glyphs for letter '{letter}'")

    out_dir = Path(output_base) / f"glyph_{letter.upper()}" / "test"
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(all_files):
        name = f"img_{idx:03d}"

        # Load image
        img = Image.open(src)

        # Convert to grayscale for Otsu, then to RGB for saving
        if img.mode == "1":
            # 1-bit: convert to uint8 grayscale
            img_gray = np.array(img.convert("L"))
        elif img.mode == "L":
            img_gray = np.array(img)
        elif img.mode in ("RGB", "RGBA"):
            img_gray = np.array(img.convert("L"))
        else:
            img_gray = np.array(img.convert("L"))

        # Resize if needed
        if size is not None:
            img_gray = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_LINEAR)

        # Save RGB image
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(images_dir / f"{name}.png"), img_rgb)

        # Compute and save Otsu mask
        mask = otsu_mask(img_gray)
        cv2.imwrite(str(masks_dir / f"{name}_mask.png"), mask)

    print(f"Written to {out_dir}")
    print(f"Run with: uv run train.py --data_folder {out_dir.parent.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--letter", type=str, default="T", help="Letter to prepare (e.g. T)")
    parser.add_argument("--source_dir", type=str,
                        default="../pnp/data/leviathan_matching_test_set_preprocessed",
                        help="Path to pnp leviathan test set directory")
    parser.add_argument("--output_base", type=str,
                        default="./data/glyph_sets",
                        help="Output base directory")
    parser.add_argument("--n", type=int, default=None,
                        help="Max number of glyphs to use (random subsample). Default: use all.")
    parser.add_argument("--size", type=int, default=None,
                        help="Resize images to size x size pixels. Default: keep original.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    prepare_glyphs(
        letter=args.letter,
        source_dir=args.source_dir,
        output_base=args.output_base,
        n=args.n,
        size=args.size,
        seed=args.seed,
    )
