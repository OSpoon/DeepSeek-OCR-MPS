import logging
import os
import re  # Import thư viện regular expression
import tempfile

import gradio as gr  # type: ignore
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logger.info("Loading model and tokenizer...")

# --- 1. Load Model and Tokenizer (Done only once at startup) ---
model_name = "./model"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Load the model to CPU first; it will be moved to GPU during processing
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval()
logger.info("✅ Model loaded successfully.")


# --- Helper function to find pre-generated result images ---
def find_result_image(path: str) -> Image.Image | None:
    for filename in os.listdir(path):
        if "grounding" in filename or "result" in filename:
            try:
                image_path = os.path.join(path, filename)
                return Image.open(image_path)
            except Exception as e:
                logger.error(f"Error opening result image {filename}: {e}")
    return None


# --- 2. Main Processing Function (UPDATED for multi-bbox drawing) ---
def process_ocr_task(
    image: Image.Image, model_size: str, task_type: str, ref_text: str
) -> tuple[str, Image.Image | None]:
    """
    Processes an image with DeepSeek-OCR for all supported tasks.
    Now draws ALL detected bounding boxes for ANY task.
    """
    if image is None:
        return "Please upload an image first.", None

    logger.info("🚀 Moving model to MPS...")
    model_mps = model.eval().to(device).to(torch.float32)
    logger.info("✅ Model is on MPS.")

    with tempfile.TemporaryDirectory() as output_path:
        # Build the prompt... (same as before)
        if task_type == "📝 Free OCR":
            prompt = "<image>\nFree OCR."
        elif task_type == "📄 Convert to Markdown":
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        elif task_type == "📈 Parse Figure":
            prompt = "<image>\nParse the figure."
        elif task_type == "🔍 Locate Object by Reference":
            if not ref_text or ref_text.strip() == "":
                raise gr.Error(
                    "For the 'Locate' task, you must provide the reference text to find!"
                )
            prompt = f"<image>\nLocate <|ref|>{ref_text.strip()}<|/ref|> in the image."
        else:
            prompt = "<image>\nFree OCR."

        temp_image_path = os.path.join(output_path, "temp_image.png")
        image.save(temp_image_path)

        # Configure model size... (same as before)
        size_configs = {
            "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
            "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
            "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
            "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
            "Gundam (Recommended)": {
                "base_size": 1024,
                "image_size": 640,
                "crop_mode": True,
            },
        }
        config = size_configs.get(model_size, size_configs["Gundam (Recommended)"])

        logger.info(f"🏃 Running inference with prompt: {prompt}")
        text_result = model_mps.infer(
            tokenizer,
            prompt=prompt,
            image_file=temp_image_path,
            output_path=output_path,
            base_size=config["base_size"],
            image_size=config["image_size"],
            crop_mode=config["crop_mode"],
            save_results=True,
            test_compress=True,
            eval_mode=True,
        )

        logger.info(f"====\n📄 Text Result: {text_result}\n====")

        # --- NEW LOGIC: Always try to find and draw all bounding boxes ---
        result_image_pil = None

        # Define the pattern to find all coordinates like [[280, 15, 696, 997]]
        pattern = re.compile(
            r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>"
        )
        matches = list(pattern.finditer(text_result))  # Use finditer to get all matches

        if matches:
            logger.info(
                f"✅ Found {len(matches)} bounding box(es). Drawing on the original image."
            )

            # Create a copy of the original image to draw on
            image_with_bboxes = image.copy()
            draw = ImageDraw.Draw(image_with_bboxes)
            w, h = image.size  # Get original image dimensions

            for match in matches:
                # Extract coordinates as integers
                coords_norm = [int(c) for c in match.groups()]
                x1_norm, y1_norm, x2_norm, y2_norm = coords_norm

                # Scale the normalized coordinates (from 1000x1000 space) to the image's actual size
                x1 = int(x1_norm / 1000 * w)
                y1 = int(y1_norm / 1000 * h)
                x2 = int(x2_norm / 1000 * w)
                y2 = int(y2_norm / 1000 * h)

                # Draw the rectangle with a red outline, 3 pixels wide
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            result_image_pil = image_with_bboxes
        else:
            # If no coordinates are found in the text, fall back to finding a pre-generated image
            logger.warning(
                "⚠️ No bounding box coordinates found in text result. Falling back to search for a result image file."
            )
            result_image_pil = find_result_image(output_path)

        return text_result, result_image_pil


# --- 3. Build the Gradio Interface (UPDATED) ---
with gr.Blocks(title="🐳DeepSeek-OCR🐳", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # DeepSeek-OCR

        **💡 How to use:**
        1.  **Upload an image** using the upload box.
        2.  Select a **Resolution**. `Gundam` is recommended for most documents.
        3.  Choose a **Task Type**:
            - **📝 Free OCR**: Extracts raw text from the image.
            - **📄 Convert to Markdown**: Converts the document into Markdown, preserving structure.
            - **📈 Parse Figure**: Extracts structured data from charts and figures.
            - **🔍 Locate Object by Reference**: Finds a specific object/text.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", label="🖼️ Upload Image", sources=["upload", "clipboard"]
            )
            model_size = gr.Dropdown(
                choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"],
                value="Gundam (Recommended)",
                label="⚙️ Resolution Size",
            )
            task_type = gr.Dropdown(
                choices=[
                    "📝 Free OCR",
                    "📄 Convert to Markdown",
                    "📈 Parse Figure",
                    "🔍 Locate Object by Reference",
                ],
                value="📄 Convert to Markdown",
                label="🚀 Task Type",
            )
            ref_text_input = gr.Textbox(
                label="📝 Reference Text (for Locate task)",
                placeholder="e.g., the teacher, 20-10, a red car...",
                visible=False,
            )
            submit_btn = gr.Button("Process Image", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="📄 Text Result", lines=15, show_copy_button=True
            )
            output_image = gr.Image(label="🖼️ Image Result (if any)", type="pil")

    # --- UI Interaction Logic ---
    def toggle_ref_text_visibility(task: str) -> gr.Textbox:
        return (
            gr.Textbox(visible=True)
            if task == "🔍 Locate Object by Reference"
            else gr.Textbox(visible=False)
        )

    task_type.change(
        fn=toggle_ref_text_visibility, inputs=task_type, outputs=ref_text_input
    )
    submit_btn.click(
        fn=process_ocr_task,
        inputs=[image_input, model_size, task_type, ref_text_input],
        outputs=[output_text, output_image],
    )

# --- 4. Launch the App ---
if __name__ == "__main__":
    if not os.path.exists("examples"):
        os.makedirs("examples")
    demo.launch()
