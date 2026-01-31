#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from PIL import Image


def center_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))


def process_image(path: Path, output_path: Path, size: int) -> None:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = center_crop(img)
        img = img.resize((size, size), Image.BICUBIC)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG", optimize=True)


def gather_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths: list[Path] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in exts:
                paths.append(path)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dataset for SDXL LoRA training."
    )
    parser.add_argument("--input_dir", required=True, help="Raw images root.")
    parser.add_argument("--output_dir", required=True, help="Processed images root.")
    parser.add_argument("--size", type=int, default=1024, help="Square size.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images = gather_images(input_dir)

    if not images:
        raise SystemExit("No images found in input_dir.")

    for path in images:
        rel = path.relative_to(input_dir)
        output_path = output_dir / rel.with_suffix(".png")
        process_image(path, output_path, args.size)

    print(f"Processed {len(images)} images to {output_dir}")


if __name__ == "__main__":
    main()
