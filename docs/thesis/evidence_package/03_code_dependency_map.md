# 03 Code Dependency Map

建立時間：2026-07-23T04:33:24.534294+00:00

## 正式 Full-image Demo 呼叫鏈

1. `run_project3_demo.ps1` 切到 project-3，啟動 `app_full_image_multilabel_ollama_gradio.py`，指定 `--project-root`、Ollama endpoint 與 Gradio port。
2. 主程式匯入三個後端模組：`src/full_image_multilabel_inference_service.py`、`src/full_image_multilabel_ollama_service.py`、`src/full_image_multilabel_report_prompt.py`。
3. Gradio `analyze_event` 接收 7 inputs，generator 每次 yield 14 outputs；`callback_validation_report.json` 驗證 no-image、validation image、GT miss、Ollama unavailable fallback 都維持 14 outputs。
4. 推論服務載入正式 checkpoint 與 threshold：
   - model：`C:\Users\09688\thoracic-cxr-project-3\outputs\full_image_224_multilabel_seed42\phase2_patch_transfer\checkpoints\full_image_multilabel_patch_transfer.pt`
   - model SHA256：`0287fe36d3623ccdb5aa43857db1168a1598788071ebdecbc43324a6953f426f`
   - thresholds：`C:\Users\09688\thoracic-cxr-project-3\outputs\full_image_224_multilabel_seed42\phase2_patch_transfer\validation_selected_thresholds.json`
   - threshold SHA256：`73a54c9b6a3de2b2f63479b0bd918836cedb577a29255b0f6e0b30dac310e9d5`
5. `predict()` 讀單張圖、轉 RGB、resize 224、ImageNet normalization，ConvNeXt-Tiny 輸出 logits `[1,5]`，再使用 `torch.sigmoid` 與 validation thresholds 得到 multi-label vector。
6. Ground Truth catalog lookup 在推論後執行，不輸入模型；`ollama_payload()` 不包含 GT 欄位。
7. Ollama 只接收 structured model result 與 patient info；`validate_response()` 要求保守語氣與固定 disclaimer。
8. Session persistence 輸出 prediction/GT/session audit 與 Markdown/PDF 報告。

## 研究訓練依賴圖

```text
raw CXR + annotation CSV
  -> BBox/ROI crop and ROI 224 manifest
  -> balanced ROI 4725 manifest (945/class)
  -> RAD-DINO CLS cache [4725,768]
  -> RAD-DINO Patch cache [4725,768,7,7]
  -> ConvNeXt-Tiny CLS distillation
  -> ConvNeXt-Tiny Patch distillation
  -> ROI grouped split
  -> ROI Phase2 ImageNet/CLS/Patch classification
  -> three-model comparison + source-cluster bootstrap
  -> Full-image multilabel 590-image split
  -> ROI Patch Proposed checkpoint transfer
  -> Full-image BCE/Sigmoid fine-tuning + validation thresholds
  -> Full-image inference service
  -> Gradio + Ground Truth + Ollama + PDF
```

## 隔離關係

- RAD-DINO 是 teacher/cache 來源，不是最終 Demo 推論模型。
- ROI Phase 2 是 five-class single-label Softmax/CrossEntropyLoss。
- Full-image Phase 2 是 five-class multilabel Sigmoid/BCEWithLogitsLoss。
- Ground Truth 只在推論後查詢，不輸入模型，也不送給 Ollama。
