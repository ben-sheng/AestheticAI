#!/usr/bin/env python3
"""
Remove furniture (sofa, chair, table, cabinet, etc.) from training images so only
background remains. Use these "background-only" images for training; then generated
scenes won't have overlapping furniture.

Flow: raw images -> this script -> background-only images -> prepare_dataset -> train.

Use --fast (default): OpenCV inpainting, no extra download, runs in minutes.
Use --sdxl: SDXL inpainting (needs ~5GB download), better quality but slow first run.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import remove as rembg_remove


def get_furniture_mask(image: Image.Image, alpha_threshold: int = 128) -> Image.Image:
    """Use rembg to get foreground (furniture) mask. Returns L image: 255 = furniture (inpaint), 0 = keep."""
    out = rembg_remove(image)
    out = out.convert("RGBA")
    alpha = out.split()[-1]
    mask = alpha.point(lambda a: 255 if a >= alpha_threshold else 0, mode="L")
    return mask


def inpaint_opencv(img: Image.Image, mask: Image.Image, radius: int = 5) -> Image.Image:
    """Fill furniture region using OpenCV inpainting (no download, fast)."""
    import cv2
    img_np = np.array(img.convert("RGB"))[:, :, ::-1]
    mask_np = np.array(mask)
    result = cv2.inpaint(img_np, mask_np, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(result[:, :, ::-1].astype("uint8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove furniture from training images; output background-only for LoRA training."
    )
    parser.add_argument("--input_dir", required=True, help="Raw training images (with furniture).")
    parser.add_argument("--output_dir", required=True, help="Where to save background-only images.")
    parser.add_argument(
        "--sdxl",
        action="store_true",
        help="Use SDXL inpainting (~5GB download). Default is OpenCV inpainting (no download).",
    )
    parser.add_argument("--model_id", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--prompt", default="minimalist European living room, empty room, clean floor and wall, no furniture, soft light")
    parser.add_argument("--negative_prompt", default="furniture, sofa, chair, table, cabinet, person, blurry")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--alpha_threshold", type=int, default=128, help="Rembg alpha above this = furniture.")
    parser.add_argument("--resolution", type=int, default=1024, help="Resize image to this (sdxl only).")
    parser.add_argument("--inpaint_radius", type=int, default=7, help="OpenCV inpaint radius (--fast only).")
    args = parser.parse_args()

    use_sdxl = args.sdxl
    if not use_sdxl:
        print("Using OpenCV inpainting (no download). Pass --sdxl for SDXL inpainting.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not images:
        raise SystemExit("No images found in input_dir.")

    pipe = None
    if use_sdxl:
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe.to(device)

    for i, path in enumerate(images):
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        print(f"[{i + 1}/{len(images)}] {path.name} -> {out_path}")

        with Image.open(path) as img:
            img = img.convert("RGB")
        w, h = img.size
        if use_sdxl and max(w, h) > args.resolution:
            scale = args.resolution / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h

        mask = get_furniture_mask(img, alpha_threshold=args.alpha_threshold)
        if mask.getextrema() == (0, 0):
            img.save(out_path)
            continue

        if use_sdxl:
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=img,
                mask_image=mask,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                strength=0.99,
            ).images[0]
        else:
            result = inpaint_opencv(img, mask, radius=args.inpaint_radius)

        result.save(out_path)

    print(f"Saved {len(images)} background-only images to {output_dir}")


if __name__ == "__main__":
    main()
