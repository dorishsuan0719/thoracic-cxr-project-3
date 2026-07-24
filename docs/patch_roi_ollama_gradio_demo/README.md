# Patch ROI Ollama Gradio Demo

本工具示範使用 RAD-DINO 7x7 Patch Distillation 初始化的 ConvNeXt-Tiny，對已裁切胸腔 X 光病灶 ROI 進行五分類，並將結構化分類結果交給本機 Ollama 產生繁體中文研究說明。

## 輸入範圍

- 僅接受預先裁切的 ROI，格式為 PNG、JPG 或 JPEG。
- 支援 Pillow 與 numpy 影像，以及 L、RGB、RGBA 模式。
- 不接受完整胸腔 X 光作為偵測輸入。
- 不執行 YOLO、BBox 預測、No finding 或臨床診斷。

## 固定模型

`outputs/raddino_convnext_tiny_patch_experiment_seed42/phase2_proposed_patch_distilled/checkpoints/patch_proposed_convnext_tiny_5class.pt`

SHA256：`8a68d68b901d721c63a38b5e75ee3291a8c06d13195572d20f29fd34a56485e5`

五個輸出類別為 Aortic enlargement、Cardiomegaly、Pleural thickening、Pulmonary fibrosis、Pleural effusion。模型使用固定 Phase 2 前處理：轉 RGB、Resize 236、Center Crop 224、雙線性插值、antialias、ImageNet mean/std。

## 啟動

先啟動 Ollama，再執行：

```powershell
python app_patch_roi_ollama_gradio.py --ollama-model gemma3:4b
```

本機網址為 `http://127.0.0.1:7860`。服務固定使用 `127.0.0.1`、`share=False`，且沒有公開推論 API。若 7860 已被占用，程式會回報 PID 並改用 7861。

完整 dry-run：

```powershell
python app_patch_roi_ollama_gradio.py --project-root C:\Users\09688\thoracic-cxr-project-3 --model C:\Users\09688\thoracic-cxr-project-3\outputs\raddino_convnext_tiny_patch_experiment_seed42\phase2_proposed_patch_distilled\checkpoints\patch_proposed_convnext_tiny_5class.pt --output-dir C:\Users\09688\thoracic-cxr-project-3\outputs\patch_roi_ollama_gradio_demo --device auto --ollama-base-url http://127.0.0.1:11434 --ollama-model auto --ollama-timeout 120 --server-name 127.0.0.1 --server-port 7860 --dry-run
```

## 安全邊界

Ollama 只收到類別、Softmax 分數、選填 Ground Truth 與模型稽核欄位，不會收到圖片或 base64。若 Ollama 離線或輸出不合規，分類結果仍保留，系統不使用雲端或其他模型替代。

本結果僅供研究與系統展示，不是臨床診斷，不能取代合格醫療專業人員的判讀。
