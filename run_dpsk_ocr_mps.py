import logging

import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "./model"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
)

model = model.eval().to(device).to(torch.float32)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = "./formula.jpg"
output_path = "./output"


# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True,
)
logger.info("Inference completed.")
logger.info(f"Generated text: {res}")
