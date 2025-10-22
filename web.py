import logging
import os
import tempfile

import gradio as gr  # type: ignore
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading model and tokenizer...")

# --- 1. Load Model and Tokenizer (Done only once at startup) ---
model_name = "./model"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager",  # Mac/MPS 必须用 eager
    trust_remote_code=True,
    use_safetensors=True,
)
# 启动时就把模型移到 MPS/CPU 并转换为 float32，避免每次请求重复加载
model = model.eval().to(device).to(torch.float32)
logger.info("✅ Model loaded successfully and moved to %s.", device)


# --- 2. Main Processing Function (完善后版本：结合简洁输出 + 多任务 + 边界框绘制) ---
def process_ocr_task(
    image: Image.Image,
    model_size: str,
    task_type: str,
    ref_text: str,
    is_eval_mode: bool,
) -> tuple[Image.Image | None, str, str]:
    """
    处理图像并返回多种输出格式：
    - 带边界框的标注图像（若检测到坐标）
    - Markdown 内容（若任务生成）
    - 纯文本 OCR 结果
    """
    if image is None:
        return None, "Please upload an image first.", "Please upload an image first."

    try:
        logger.info("🚀 Running inference on %s...", device)
        # 模型已在启动时移到 device，直接使用

        with tempfile.TemporaryDirectory() as output_path:
            # 根据任务类型设置 prompt
            if task_type == "📝 Free OCR":
                prompt = "<image>\nFree OCR."
            elif task_type == "📄 Convert to Markdown":
                prompt = "<image>\n<|grounding|>Convert the document to markdown."
            elif task_type == "📈 Parse Figure":
                prompt = "<image>\nParse the figure."
            elif task_type == "🔍 Locate Object by Reference":
                if not ref_text or ref_text.strip() == "":
                    return (
                        None,
                        "❌ Reference text is required for Locate task.",
                        "❌ Reference text is required for Locate task.",
                    )
                prompt = (
                    f"<image>\nLocate <|ref|>{ref_text.strip()}<|/ref|> in the image."
                )
            else:
                prompt = "<image>\nFree OCR."

            # 保存上传的图像
            temp_image_path = os.path.join(output_path, "temp_image.png")
            image.save(temp_image_path)

            # 配置模型尺寸参数
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

            logger.info("🏃 Running inference with prompt: %s", prompt)
            plain_text_result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=output_path,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=True,
                test_compress=True,
                eval_mode=is_eval_mode,
            )

            logger.info(
                "📄 Text result (length=%d)",
                len(plain_text_result) if plain_text_result else 0,
            )

            # 读取生成的 markdown 文件（如果存在）
            markdown_result_path = os.path.join(output_path, "result.mmd")
            markdown_content = ""
            if os.path.exists(markdown_result_path):
                with open(markdown_result_path, encoding="utf-8") as f:
                    markdown_content = f.read()
            else:
                markdown_content = "Markdown result was not generated. This is expected for 'Free OCR' task."

            # 读取模型自动生成的标注图像（若存在）
            result_image_pil = None
            image_result_path = os.path.join(output_path, "result_with_boxes.jpg")
            if os.path.exists(image_result_path):
                result_image_pil = Image.open(image_result_path)
                result_image_pil.load()
                logger.info("✅ Found annotated image: %s", image_result_path)
            else:
                logger.info("ℹ️ No annotated image generated (expected for some tasks).")

            # 返回三个输出：标注图像、Markdown 预览内容、纯文本结果
            text_result = plain_text_result if plain_text_result else markdown_content
            return result_image_pil, markdown_content, text_result

    except Exception as e:
        logger.exception("❌ Inference failed")
        error_msg = f"ERROR: {e}"
        return None, error_msg, error_msg


# --- 3. Build the Gradio Interface (完善后版本：Tab 展示 + eval_mode 控制) ---
with gr.Blocks(title="🐳 DeepSeek-OCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐳 DeepSeek-OCR Demo (Mac/MPS 适配版)

        **💡 使用说明：**
        1. **上传图片** 到上传框。
        2. 选择 **分辨率**。推荐使用 `Gundam` 以获得文档最佳效果。
        3. 选择 **任务类型**：
            - **📝 Free OCR**：提取原始文本
            - **📄 Convert to Markdown**：转换为 Markdown 格式（保留结构）
            - **📈 Parse Figure**：提取图表结构化数据
            - **🔍 Locate Object by Reference**：定位特定对象/文本（需填写参考文本）
        4. 可选：勾选 **Evaluation Mode** 以仅返回纯文本（可能更快，但不生成标注图像和 Markdown）。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="🖼️ 上传图片")

            model_size = gr.Dropdown(
                choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"],
                value="Gundam (Recommended)",
                label="⚙️ 分辨率大小",
            )

            task_type = gr.Dropdown(
                choices=[
                    "📝 Free OCR",
                    "📄 Convert to Markdown",
                    "📈 Parse Figure",
                    "🔍 Locate Object by Reference",
                ],
                value="📄 Convert to Markdown",
                label="🚀 任务类型",
            )

            ref_text_input = gr.Textbox(
                label="📝 参考文本（用于 Locate 任务）",
                placeholder="例如: the teacher, 20-10, a red car...",
                visible=False,
            )

            eval_mode_checkbox = gr.Checkbox(
                value=False,
                label="启用 Evaluation Mode",
                info="仅返回纯文本，可能更快。取消勾选可获得标注图像和 Markdown。",
            )

            submit_btn = gr.Button("🚀 处理图片", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📷 标注图像"):
                    output_image = gr.Image(
                        interactive=False, label="带边界框的图像（如有检测到）"
                    )
                with gr.TabItem("📝 Markdown 预览"):
                    output_markdown = gr.Markdown(label="Markdown 渲染预览")
                with gr.TabItem("📄 Markdown 源码 / 纯文本输出"):
                    output_text = gr.Textbox(
                        lines=20,
                        show_copy_button=True,
                        interactive=False,
                        label="纯文本结果或 Markdown 源码",
                    )

    # --- UI 交互逻辑：根据任务类型显示/隐藏参考文本框 ---
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
        inputs=[image_input, model_size, task_type, ref_text_input, eval_mode_checkbox],
        outputs=[output_image, output_markdown, output_text],
    )

# --- 4. Launch the App ---
if __name__ == "__main__":
    if not os.path.exists("examples"):
        os.makedirs("examples")
    demo.launch()
