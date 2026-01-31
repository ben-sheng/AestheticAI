#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

# #region agent log
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = ""):
    import time
    log_path = Path(__file__).resolve().parent.parent / ".cursor" / "debug.log"
    payload = {"location": location, "message": message, "data": data, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "hypothesisId": hypothesis_id}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
# #endregion

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig


# =========================
# Dataset
# =========================
class ImagePromptDataset(Dataset):
    def __init__(self, data_dir: Path, prompt: str, size: int):
        self.images = sorted(
            p for p in data_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not self.images:
            raise RuntimeError("No images found")

        self.prompt = prompt
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with Image.open(self.images[idx]) as img:
            img = img.convert("RGB")
        return {
            "pixel_values": self.transform(img),
            "prompt": self.prompt,
        }


# =========================
# Training
# =========================
def main():
    parser = argparse.ArgumentParser("Train SDXL LoRA (diffusers 0.36 stable)")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--instance_prompt", required=True)

    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    # GradScaler expects FP32 params; load in FP32 when using fp16 AMP so autocast handles cast in forward only.
    load_dtype = torch.float32 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)

    # =========================
    # Load SDXL pipeline
    # =========================
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=load_dtype,
        use_safetensors=True,
        variant=None,
    ).to(device)

    vae = pipe.vae
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2

    # Freeze base model
    for m in [vae, text_encoder, text_encoder_2, unet]:
        m.requires_grad_(False)
        m.eval()

    # =========================
    # LoRA (PEFT)
    # =========================
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )

    unet.add_adapter(lora_config)
    unet.train()

    # #region agent log
    trainable = [p for p in unet.parameters() if p.requires_grad]
    _debug_log("train_lora_sdxl.py:after_add_adapter", "trainable param dtypes", {"count": len(trainable), "dtypes": [str(p.dtype) for p in trainable[:5]], "first_param_dtype": str(trainable[0].dtype) if trainable else None}, "B")
    _debug_log("train_lora_sdxl.py:after_add_adapter", "unet dtype", {"mixed_precision": args.mixed_precision, "dtype_arg": str(dtype)}, "A")
    # #endregion

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    # #region agent log
    opt_param_dtypes = [str(p.dtype) for grp in optimizer.param_groups for p in grp["params"]]
    _debug_log("train_lora_sdxl.py:after_optimizer", "optimizer param dtypes", {"param_dtypes": opt_param_dtypes[:5], "all_same": len(set(opt_param_dtypes)) == 1}, "A")
    # #endregion

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    dataset = ImagePromptDataset(
        Path(args.data_dir),
        args.instance_prompt,
        args.resolution,
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision == "fp16")

    # =========================
    # Training loop
    # =========================
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            prompts = batch["prompt"]

            # ---- Encode images
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents *= vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # ---- Encode text (关键：不要用位置解包)
            text_outputs = pipe.encode_prompt(
                prompts,
                device=device,
                do_classifier_free_guidance=False,
            )

            # SDXL 在 diffusers 0.36 中的稳定取法
            prompt_embeds = text_outputs[0]
            pooled_prompt_embeds = text_outputs[-2]

            # ---- SDXL additional conditioning
            add_time_ids = pipe._get_add_time_ids(
                (args.resolution, args.resolution),
                (0, 0),
                (args.resolution, args.resolution),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
            ).to(device)
            add_time_ids = add_time_ids.repeat(latents.shape[0], 1)

            with torch.amp.autocast("cuda", enabled=args.mixed_precision == "fp16"):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            scaler.scale(loss).backward()
            # #region agent log
            grad_dtypes = []
            for grp in optimizer.param_groups:
                for p in grp["params"]:
                    if p.grad is not None:
                        grad_dtypes.append(str(p.grad.dtype))
                        break
                if grad_dtypes:
                    break
            _debug_log("train_lora_sdxl.py:before_scaler_step", "grad dtypes before scaler.step", {"grad_dtypes_sample": grad_dtypes, "scaler_enabled": args.mixed_precision == "fp16"}, "C")
            # #endregion
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 50 == 0:
                print(f"step {global_step}/{args.max_train_steps} - loss {loss.item():.4f}")

            if global_step >= args.max_train_steps:
                break

    # =========================
    # Save LoRA
    # =========================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unet.save_attn_procs(output_dir)
    with (output_dir / "train_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"✅ Saved LoRA weights to {output_dir}")


if __name__ == "__main__":
    main()
