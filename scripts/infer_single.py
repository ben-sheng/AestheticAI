#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from PIL import Image

from diffusers import StableDiffusionXLInpaintPipeline


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGBA")


def make_mask_from_alpha(rgba: Image.Image) -> Image.Image:
    alpha = rgba.split()[-1]
    mask = alpha.point(lambda a: 255 if a == 0 else 0)
    return mask.convert("L")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single branded background image.")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA weights directory.")
    parser.add_argument("--input_image", required=True, help="Furniture image (PNG with alpha recommended).")
    parser.add_argument("--mask_image", help="Optional mask image (white=inpaint, black=keep).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    image_rgba = load_image(Path(args.input_image))
    image_rgba = image_rgba.resize((args.resolution, args.resolution), Image.BICUBIC)

    if args.mask_image:
        mask = Image.open(args.mask_image).convert("L").resize(
            (args.resolution, args.resolution), Image.NEAREST
        )
    else:
        mask = make_mask_from_alpha(image_rgba)

    base_image = Image.new("RGB", image_rgba.size, (255, 255, 255))
    base_image.paste(image_rgba, mask=image_rgba.split()[-1])

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    pipe.to(device)
    pipe.unet.load_attn_procs(args.lora_path)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    ).images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
