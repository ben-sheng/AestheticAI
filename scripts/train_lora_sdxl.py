#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0

class ImagePromptDataset(Dataset):
    def __init__(self, data_dir: Path, prompt: str, size: int) -> None:
        self.images = sorted(
            [
                p
                for p in data_dir.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
        )
        if not self.images:
            raise ValueError("No images found in dataset.")
        self.prompt = prompt
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        path = self.images[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return {"pixel_values": self.transform(image), "prompt": self.prompt}


def add_lora_to_unet(unet, rank: int) -> None:
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"Unexpected layer name: {name}")

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            lora_attn_procs[name] = LoRAAttnProcessor2_0(
                rank=rank
            )
        else:
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            )

    unet.set_attn_processor(lora_attn_procs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SDXL LoRA on a small furniture dataset.")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--instance_prompt", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        variant="fp16" if args.mixed_precision == "fp16" else None,
        use_safetensors=True,
    )
    pipe.to(device)

    vae = pipe.vae
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2

    for module in [vae, text_encoder, text_encoder_2]:
        module.requires_grad_(False)
        module.eval()

    unet.requires_grad_(False)
    add_lora_to_unet(unet, args.rank)
    for _, proc in unet.attn_processors.items():
        for param in proc.parameters():
            param.requires_grad_(True)
    unet.train()

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    dataset = ImagePromptDataset(Path(args.data_dir), args.instance_prompt, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    total_steps = args.max_train_steps
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")

    while global_step < total_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            prompts = batch["prompt"]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(
                prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

            add_time_ids = pipe._get_add_time_ids(
                (args.resolution, args.resolution),
                (0, 0),
                (args.resolution, args.resolution),
                dtype=prompt_embeds.dtype,
            ).to(device)
            add_time_ids = add_time_ids.repeat(latents.shape[0], 1)

            with torch.cuda.amp.autocast(enabled=args.mixed_precision == "fp16"):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                ).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 50 == 0:
                print(f"step {global_step}/{total_steps} - loss {loss.item():.4f}")
            if global_step >= total_steps:
                break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    unet.save_attn_procs(output_dir)
    print(f"Saved LoRA weights to {output_dir}")


if __name__ == "__main__":
    main()
