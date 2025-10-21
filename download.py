import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

snapshot_download("deepseek-ai/DeepSeek-OCR", local_dir="./model")
