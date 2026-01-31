AestheticAI (Hackathon)

Goal
- Train a small SDXL LoRA that captures a furniture brand's visual style.
- Generate a single image from a furniture upload with a brand-matching background.

Quick start
1) Create and activate a Python venv.
2) Install requirements: pip install -r requirements.txt
3) Prepare data (see data/README.md).
4) Train: python scripts/train_lora_sdxl.py --help
5) Inference: python scripts/infer_single.py --help

Notes
- Training and inference are intended to run on a local RTX 4090.
- The web demo is intentionally deferred; focus is the model first.
