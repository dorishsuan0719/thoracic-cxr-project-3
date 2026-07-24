# 技術稽核

## 模型

- Checkpoint SHA256：`8a68d68b901d721c63a38b5e75ee3291a8c06d13195572d20f29fd34a56485e5`
- Architecture：ConvNeXt-Tiny
- Initialization：RAD-DINO 7x7 patch distilled
- 完整 export state dict：182 keys
- Strict load：PASS
- Missing keys：0
- Unexpected keys：0
- `model.eval()`：是
- Trainable parameters：0
- 推論區段：`torch.inference_mode()`
- Optimizer：未建立
- Backward：未呼叫
- Singleton：double-check lock
- 同時推論：inference lock

## 前處理

直接重用 `src/infer_patch_proposed_single_roi.py` 匯出的 `Phase2Transform(training=False)`。固定流程為 RGB、Resize 236、Center Crop 224、BILINEAR、antialias、tensor 與 ImageNet mean/std。沒有 augmentation 或 TTA。

## 本機 Ollama

- Base URL：`http://127.0.0.1:11434`
- `/api/tags`：PASS
- 驗證模型：`gemma3:4b`
- `/api/chat`：PASS
- 圖片送入 Ollama：否
- 離線分類保留測試：PASS

## Smoke Test

- Validation ROI：5 張，class 0 一張、class 2 兩張、class 4 兩張
- Test images read：0
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- Model load count：1
- Inference request count：5
- NaN：0
- Inf：0
- 實際 Ollama generation：PASS，57.94 秒
- 模擬 Ollama 離線：PASS，分類輸出仍保留
- `inference_history.csv`：5 列，UTF-8 BOM
- `ollama_history.csv`：2 列，UTF-8 BOM

## 受保護來源雜湊

- `src/infer_patch_proposed_single_roi.py`：`8c73d3c4b5268d3783107b3e173943b741387d413e9632901179e3a3d0c7600c`
- Validation manifest：`5f92fd7282df28a4ec3365ba5fa7a777b365db860f7991a47238162d1ac5bc00`
- Test manifest：`2130a73dcbadec1d6b4bba68f809db7eeed25d1ea421c4d450d3e0b4d015551a`
- `selected_validation_samples.csv`：`19f8fe383a38ed76be85978c085797b79e39b517e63f978f9f27ca53d1fc2352`
- `comparison_summary.json`：`0950224382d6b29d99d892b5fbe670d7d18bb7b19fadc35c36a99521c33f574b`
- `final_research_report.md`：`1e3ae2c0ab83ae990ac3f13d98cf647296ebf51571170291754700da8ec83cde`

## 服務邊界

Gradio 綁定 `127.0.0.1`，`share=False`、`show_error=False`，事件 API 設為 private，沒有公開分享網址或公開推論 API。系統不載入 RAD-DINO teacher、CLS Proposed、Baseline 或 Test 圖片。
