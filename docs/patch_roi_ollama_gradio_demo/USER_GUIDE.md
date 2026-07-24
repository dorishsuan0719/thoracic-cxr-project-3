# 使用指南

## 啟動順序

1. 確認本機 Ollama 正在 `http://127.0.0.1:11434` 執行。
2. 確認指定模型已安裝，詳見 `OLLAMA_SETUP.md`。
3. 從專案根目錄執行 `python app_patch_roi_ollama_gradio.py --ollama-model gemma3:4b`。
4. 開啟 `http://127.0.0.1:7860`。

## 操作

1. 上傳一張已裁切病灶 ROI。不要上傳完整胸腔 X 光影像。
2. Ground Truth 可留空；若是已知 Validation ROI，可選擇正確類別。
3. 按下「執行分類與本機說明」。
4. 查看預測類別、Softmax confidence、五類分布、排序與模型稽核。
5. Ground Truth 已提供時，頁面會顯示 true-class probability 與 Correct/Incorrect。
6. Ollama 說明只解釋固定模型輸出，不會看圖或重新分類。

## 輸入警告

- 非 224x224 ROI 仍可執行，但會顯示 Resize 236 與 Center Crop 224 警告。
- 長寬比超過 3 時會提醒確認 Ground Truth BBox 裁切是否合理。
- 空白圖、零尺寸圖、無法解碼圖片及不支援格式會被拒絕。

## 常見狀況

- Ollama 說明失敗：分類結果仍有效；確認 Ollama 服務、模型名稱與 timeout。
- 7860 被占用：終端會列出 PID，服務改用 `http://127.0.0.1:7861`。
- CUDA 無法使用：`--device auto` 會使用可用裝置；也可明確指定 `--device cpu`。
- 詳細例外只寫入 `outputs/patch_roi_ollama_gradio_demo/logs/error.log`，介面不顯示 traceback。

本結果僅供研究與系統展示，不是臨床診斷，不能取代合格醫療專業人員的判讀。
