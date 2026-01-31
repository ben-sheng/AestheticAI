Usage

Data preparation
1) Place raw images in: data/raw/<brand_name>/
2) Run:
   python scripts/prepare_dataset.py --input_dir data/raw/<brand_name> --output_dir data/processed/<brand_name>

Training
Example:
python scripts/train_lora_sdxl.py \
  --data_dir data/processed/<brand_name> \
  --output_dir models/<brand_name>_lora \
  --instance_prompt "a photo of <brand_token> furniture" \
  --max_train_steps 800 \
  --learning_rate 1e-4 \
  --train_batch_size 1

Notes
- Use a stable, short brand token in the prompt (for example: "skf").
- Keep the prompt consistent across training and inference.
- Increase max_train_steps if the style is not strong enough.

Inference (single output image)
If your furniture PNG has transparency, the alpha channel is used as mask.
Otherwise, provide a mask image where white pixels are background.

Example:
python scripts/infer_single.py \
  --lora_path models/<brand_name>_lora \
  --input_image inputs/chair.png \
  --prompt "a studio background, soft daylight, <brand_token> style" \
  --output outputs/chair_with_background.png
