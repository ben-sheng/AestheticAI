#!/usr/bin/env python3
"""
Generate final images using the trained LoRA.
For each image in the test folder, generates one image with the LoRA prompt
and saves to the output directory.
"""
import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


def get_instance_prompt_from_config(lora_path: Path) -> str | None:
    """Load instance_prompt from train_config.json if present."""
    config_path = lora_path / "train_config.json"
    if not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return config.get("instance_prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images with trained LoRA using test folder images as count/names.")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA weights directory (e.g. models/woltu_lora).")
    parser.add_argument("--test_dir", default="test", help="Folder containing test images (used for count and output names).")
    parser.add_argument("--output_dir", default="test/output", help="Where to save generated images.")
    parser.add_argument("--prompt", help="Override prompt; if not set, uses instance_prompt from train_config.json or a default.")
    parser.add_argument("--negative_prompt", default="blurry, low quality, distorted")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    lora_path = Path(args.lora_path)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    test_images = sorted(
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )
    if not test_images:
        raise RuntimeError(f"No images found in {test_dir}")

    prompt = args.prompt
    if prompt is None:
        prompt = get_instance_prompt_from_config(lora_path) or "a photo of woltu furniture"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.to(device)
    pipe.unet.load_attn_procs(lora_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    for i, input_path in enumerate(test_images):
        stem = input_path.stem
        out_path = output_dir / f"{stem}_generated.png"
        print(f"[{i + 1}/{len(test_images)}] Generating for {input_path.name} -> {out_path}")

        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
        ).images[0]

        image.save(out_path)

    print(f"Saved {len(test_images)} image(s) to {output_dir}")


if __name__ == "__main__":
    main()
