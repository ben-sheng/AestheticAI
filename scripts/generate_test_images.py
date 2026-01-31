#!/usr/bin/env python3
"""
Generate final images using the trained LoRA.
- Default: generate scene from prompt only.
- With --composite: keep your test images (matting: remove white edges), generate scene with LoRA,
  then paste the cut-out product onto the generated scene.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline


def remove_white_background(img: Image.Image, threshold: int = 248) -> Image.Image:
    """Remove white/near-white background; return RGBA with subject only (抠图去白边)."""
    img = img.convert("RGB")
    w, h = img.size
    data = img.getdata()
    alpha = []
    for (r, g, b) in data:
        if r >= threshold and g >= threshold and b >= threshold:
            alpha.append(0)
        else:
            alpha.append(255)
    out = Image.new("RGBA", (w, h))
    out.putdata([(r, g, b, a) for (r, g, b), a in zip(data, alpha)])
    return out


def feather_alpha(rgba: Image.Image, radius: int = 3) -> Image.Image:
    """Soften alpha edge so product blends into scene (融入)."""
    from PIL import ImageFilter
    r, g, b, a = rgba.split()
    a_smooth = a.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.merge("RGBA", (r, g, b, a_smooth))


def _auto_placement_and_scale(
    sw: int, sh: int, pw: int, ph: int, index: int,
) -> tuple[float, int, int]:
    """Decide scale and (x,y) so product fits scene harmoniously (协调)."""
    # Scale: 0.38–0.50 of smaller scene dimension so scene is visible and product not overwhelming
    scale_min, scale_max = 0.38, 0.50
    t = (index % 5) / 5.0
    scale_max_use = scale_min + (scale_max - scale_min) * (0.7 + 0.3 * t)
    scale_by_w = (sw * scale_max_use) / pw
    scale_by_h = (sh * scale_max_use) / ph
    scale = min(scale_by_w, scale_by_h, 1.0)
    new_pw = int(pw * scale)
    new_ph = int(ph * scale)
    margin_bottom = max(sh // 15, 20)
    margin_side = max(sw // 12, 30)
    y = sh - new_ph - margin_bottom
    # Slight horizontal variation: center with small offset so not always same
    offsets = [-sw // 8, 0, sw // 8, -sw // 12, sw // 12]
    x_center = (sw - new_pw) // 2
    x = x_center + offsets[index % len(offsets)]
    x = max(margin_side, min(sw - new_pw - margin_side, x))
    return scale, x, y


def composite_product_onto_scene(
    product_rgba: Image.Image,
    scene: Image.Image,
    scale_max: float = 0.50,
    position: str = "auto",
    image_index: int = 0,
    feather: bool = True,
) -> Image.Image:
    """Place matted product onto scene; auto = decide size and position for harmony (协调融入)."""
    scene = scene.convert("RGB")
    sw, sh = scene.size
    if feather:
        product_rgba = feather_alpha(product_rgba, radius=2)
    pw, ph = product_rgba.size
    if position == "auto":
        scale, x, y = _auto_placement_and_scale(sw, sh, pw, ph, image_index)
    else:
        scale = min(sw * scale_max / pw, sh * scale_max / ph, 1.0)
        x = (sw - int(pw * scale)) // 2
        y = (sh - int(ph * scale)) // 2
        if position == "bottom_center":
            y = sh - int(ph * scale) - max(0, sh // 20)
    new_pw, new_ph = int(pw * scale), int(ph * scale)
    product_scaled = product_rgba.resize((new_pw, new_ph), Image.Resampling.LANCZOS)
    out = scene.copy()
    out.paste(product_scaled, (x, y), product_scaled)
    return out


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
    parser.add_argument("--style", default="living_room", choices=["dining_room", "living_room", "bedroom", "office", "product", "none"],
                        help="Scene style (default: living_room). Use 'product' for clean product-style like training data.")
    parser.add_argument("--negative_prompt", default="blurry, low quality, distorted, abstract, amorphous, messy, surreal, melted, unformed, cluttered, deformed, out of focus, duplicate, ornate, baroque, busy, crowded, too many objects, chaotic, messy interior")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=8.0, help="Higher = follow prompt more (cleaner, less messy).")
    parser.add_argument("--resolution", type=int, default=1280, help="Square size when --width/--height not set.")
    parser.add_argument("--width", type=int, default=2570, help="Output width (default 2570).")
    parser.add_argument("--height", type=int, default=1276, help="Output height (default 1276).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--composite", action="store_true",
                        help="Keep test images: matting (remove white), generate scene with LoRA, paste product onto scene.")
    parser.add_argument("--white_threshold", type=int, default=248,
                        help="Pixels with R,G,B >= this become transparent (default 248).")
    parser.add_argument("--product_scale", type=float, default=0.50,
                        help="Max size of product (used when --product_position is not auto).")
    parser.add_argument("--product_position", default="auto", choices=["auto", "center", "bottom_center"],
                        help="Placement: auto = decide size and position for harmony (default).")
    parser.add_argument("--no_feather", action="store_true", help="Disable edge feathering (hard cutout).")
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

    # Match training data: 简约 (minimalist) + 欧式 (European style) — clean, uncluttered, European interior
    quality_suffix = ", minimalist, European style, clean lines, uncluttered, sharp focus, realistic, well-defined"
    style_suffixes = {
        "dining_room": ", in a minimalist European dining room, simple elegant interior, warm lighting, few objects",
        "living_room": ", in a minimalist European living room, Scandinavian style, clean interior, natural light, simple composition",
        "bedroom": ", in a minimalist European bedroom, simple elegant interior, soft lighting, uncluttered",
        "office": ", in a minimalist European office, clean professional interior, simple composition",
        "product": ", minimalist European product photography, simple background, clean furniture, sharp focus",
        "none": "",
    }
    prompt = prompt.rstrip() + quality_suffix
    if style_suffixes[args.style]:
        prompt = prompt + style_suffixes[args.style]

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

    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else args.resolution
    # SDXL expects multiples of 8; round to avoid surprises
    width = (width // 8) * 8
    height = (height // 8) * 8

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    for i, input_path in enumerate(test_images):
        stem = input_path.stem
        out_path = output_dir / f"{stem}_generated.png"
        print(f"[{i + 1}/{len(test_images)}] {input_path.name} -> {out_path}")

        if args.composite:
            with Image.open(input_path) as test_img:
                test_img = test_img.convert("RGB")
            product_rgba = remove_white_background(test_img, threshold=args.white_threshold)

            scene = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]

            image = composite_product_onto_scene(
                product_rgba,
                scene,
                scale_max=args.product_scale,
                position=args.product_position,
                image_index=i,
                feather=not args.no_feather,
            )
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]

        image.save(out_path)

    print(f"Saved {len(test_images)} image(s) to {output_dir}")


if __name__ == "__main__":
    main()
