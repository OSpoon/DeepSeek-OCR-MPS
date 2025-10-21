#!/usr/bin/env bash
# Minimal hotfix: download fixed file and overwrite ./model/modeling_deepseekocr.py
# https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/20
set -euo pipefail

URL="https://huggingface.co/deepseek-ai/DeepSeek-OCR/raw/1e3401a3d4603e9e71ea0ec850bfead602191ec4/modeling_deepseekocr.py"
DEST="./model/modeling_deepseekocr.py"

mkdir -p ./model
TMP="$(mktemp "./model/modeling_deepseekocr.py.tmp.XXXXXX")"
echo "[hotfix] Downloading: $URL"
curl -fL --retry 3 --connect-timeout 30 "$URL" -o "$TMP"
mv -f "$TMP" "$DEST"
echo "[hotfix] Replaced: $DEST"
ls -l "$DEST" || true
