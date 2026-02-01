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


def _auto_scale_only(
    sw: int, sh: int, pw: int, ph: int, index: int,
) -> tuple[float, int, int]:
    """Decide scale and resulting (new_pw, new_ph). Position is chosen later from scene analysis."""
    scale_min, scale_max = 0.42, 0.48
    t = (index % 5) / 5.0
    scale_max_use = scale_min + (scale_max - scale_min) * (0.8 + 0.2 * t)
    scale_by_w = (sw * scale_max_use) / pw
    scale_by_h = (sh * scale_max_use) / ph
    scale = min(scale_by_w, scale_by_h, 1.0)
    new_pw = int(pw * scale)
    new_ph = int(ph * scale)
    return scale, new_pw, new_ph


def _analyze_scene_placement(scene: Image.Image, new_pw: int, new_ph: int) -> tuple[int, int]:
    """
    Analyze background to find where the product best fits (真正融入).
    Prefer: low variance (floor/carpet), low edge density (open space), in lower half.
    """
    from PIL import ImageFilter
    sw, sh = scene.size
    gray = scene.convert("L")
    pixels = list(gray.getdata())
    # Edge map: high value = edge; we want to avoid busy areas
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_data = list(edge.getdata())
    # Grid over lower ~55% of image (floor zone), 5 cols x 2 rows
    ncols, nrows = 5, 2
    row_start = int(sh * 0.45)
    row_end = sh
    col_w = sw // ncols
    row_h = (row_end - row_start) // nrows
    best_score = -1.0
    best_cx, best_cy = sw // 2, row_start + row_h // 2
    for ri in range(nrows):
        for ci in range(ncols):
            x0 = ci * col_w
            y0 = row_start + ri * row_h
            x1 = min(x0 + col_w, sw)
            y1 = min(y0 + row_h, row_end)
            if x1 <= x0 or y1 <= y0:
                continue
            # Score: low variance + low edges = suitable floor/empty space
            var_sum = 0.0
            mean_sum = 0.0
            count = 0
            edge_sum = 0
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    idx = yy * sw + xx
                    p = pixels[idx]
                    e = edge_data[idx]
                    mean_sum += p
                    var_sum += p * p
                    edge_sum += e
                    count += 1
            if count == 0:
                continue
            mean_ = mean_sum / count
            variance = (var_sum / count) - (mean_ * mean_)
            edge_mean = edge_sum / count
            # Prefer uniform (low var) and non-busy (low edge); avoid very dark (likely shadow)
            score = 1.0 / (1.0 + variance * 0.01) * 1.0 / (1.0 + edge_mean * 0.02)
            if mean_ < 40:
                score *= 0.5
            if score > best_score:
                best_score = score
                best_cx = (x0 + x1) // 2
                best_cy = (y0 + y1) // 2
    # Place product so its bottom-center is near (best_cx, best_cy); product sits on floor
    margin_bottom = max(sh // 14, 16)
    y = sh - new_ph - margin_bottom
    x = best_cx - new_pw // 2
    x = max(0, min(sw - new_pw, x))
    return x, y


def _draw_soft_shadow(
    scene: Image.Image, px: int, py: int, pw: int, ph: int,
    shadow_alpha: int = 72, blur_radius: int = 20,
) -> Image.Image:
    """Draw a soft elliptical shadow under the product to ground it (落地感). Returns new RGB image."""
    from PIL import ImageDraw, ImageFilter
    w, h = scene.size
    scene_rgba = scene.convert("RGBA")
    pad = max(pw // 5, 24)
    sx = max(0, px - pad)
    sy = py + ph - ph // 6
    sw = min(pw + 2 * pad, w - sx)
    sh = max(ph // 3, 28)
    if sy + sh > h:
        sh = h - sy
    if sw <= 0 or sh <= 0:
        return scene
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse([sx, sy, sx + sw, sy + sh], fill=(0, 0, 0, shadow_alpha))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = Image.alpha_composite(scene_rgba, overlay)
    return out.convert("RGB")


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
        scale, new_pw, new_ph = _auto_scale_only(sw, sh, pw, ph, image_index)
        x, y = _analyze_scene_placement(scene, new_pw, new_ph)
    else:
        scale = min(sw * scale_max / pw, sh * scale_max / ph, 1.0)
        new_pw = int(pw * scale)
        new_ph = int(ph * scale)
        x = (sw - new_pw) // 2
        y = (sh - new_ph) // 2
        if position == "bottom_center":
            y = sh - new_ph - max(0, sh // 20)
    product_scaled = product_rgba.resize((new_pw, new_ph), Image.Resampling.LANCZOS)
    out = scene.copy()
    if position == "auto":
        out = _draw_soft_shadow(out, x, y, new_pw, new_ph)
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
    parser.add_argument("--negative_prompt", default="blurry, low quality, distorted, abstract, amorphous, messy, surreal, melted, unformed, cluttered, deformed, out of focus, duplicate, ornate, baroque, busy, crowded, too many objects, chaotic, messy interior, complex background, many decorations, filled shelves, lots of furniture, busy wall, detailed wallpaper, patterned floor, crowded scene, shelves, plants on wall, picture frames, vases, rugs, multiple sofas, sofa, armchairs, side tables, coffee table, cabinet, bookshelf, wall art, ornaments, knick knacks, layered decor")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="Higher = follow prompt more (simpler background).")
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

    # 背景简单不乱：空白墙、少物、简约欧式
    quality_suffix = ", minimalist European style, empty plain background, blank white or light grey wall, bare floor, no decorations, no furniture in background, no shelves, no plants, no pictures, sharp focus, realistic"
    style_suffixes = {
        "dining_room": ", empty minimalist European dining room, plain wall only, simple floor, warm light, nothing on walls",
        "living_room": ", empty minimalist European living room, blank wall, plain floor, natural light, no decor, no shelves, no plants",
        "bedroom": ", empty minimalist European bedroom, plain wall, simple floor, soft light, no decorations",
        "office": ", empty minimalist European office, blank wall, plain background, no decor",
        "product": ", minimalist European product photography, plain white or neutral wall background, sharp focus",
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
