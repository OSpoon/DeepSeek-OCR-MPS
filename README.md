# DeepSeek-OCR-MPS

在 macOS（Apple 芯片 M1/M2/M3）上以 MPS 运行 DeepSeek-OCR。

## 安装

使用你 `uv` 虚拟环境工具创建环境并安装依赖。

```zsh
uv sync
source .venv/bin/activate
```

## 下载模型

方式一：脚本下载（默认保存到 `./model`）

```zsh
python download.py
```

## 应用热修复（MPS 兼容）

本仓库提供了一个 `hotfix.sh` 脚本，会覆盖 `./model/modeling_deepseekocr.py`：

```zsh
chmod +x ./hotfix.sh
./hotfix.sh
```

PS: https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/20

## 运行推理

```zsh
python run_dpsk_ocr_mps.py
```

## 致谢

- 感谢 DeepSeek 开源模型与 Hugging Face 生态。
