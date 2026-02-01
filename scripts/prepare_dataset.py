#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from PIL import Image


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))


def resize_cover_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize to cover (target_w, target_h) then center crop."""
    w, h = image.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = image.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def process_image(path: Path, output_path: Path, size: int, width: int | None = None, height: int | None = None) -> None:
    with Image.open(path) as img:
        img = img.convert("RGB")
    if width is not None and height is not None:
        img = resize_cover_crop(img, width, height)
    else:
        img = center_crop_square(img)
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
    parser = argparse.ArgumentParser(description="Prepare dataset for SDXL LoRA training.")
    parser.add_argument("--input_dir", required=True, help="Raw images root.")
    parser.add_argument("--output_dir", required=True, help="Processed images root.")
    parser.add_argument("--size", type=int, default=1024, help="Square size when not using --width/--height.")
    parser.add_argument("--width", type=int, default=None, metavar="W", help="Output width (use with --height for 2568x1272 aspect).")
    parser.add_argument("--height", type=int, default=None, metavar="H", help="Output height (use with --width).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images = gather_images(input_dir)

    if not images:
        raise SystemExit("No images found in input_dir.")

    w, h = args.width, args.height
    for path in images:
        rel = path.relative_to(input_dir)
        output_path = output_dir / rel.with_suffix(".png")
        process_image(path, output_path, args.size, width=w, height=h)

    print(f"Processed {len(images)} images to {output_dir}")


if __name__ == "__main__":
    main()
