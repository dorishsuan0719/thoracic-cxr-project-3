# Full-image Gradio Demo 離線與故障備援指南

## 目標

本 Demo 預設以完全離線本機模式執行：`server_name=127.0.0.1`、`share=False`，不建立 `gradio.live` 公開網址，也不呼叫任何雲端 LLM API。Ollama 僅作為本機文字說明服務；影像分類永遠由本機 Full-image ConvNeXt-Tiny 多標籤模型完成。

## 無網路啟動

1. 確認正式 checkpoint、validation thresholds 與 Ground Truth catalog 已位於專案內。
2. 確認 Ollama 已安裝且本機已有 `gemma3:4b`。啟動腳本只檢查，不會自動下載模型或套件。
3. 執行：

```bat
scripts\start_demo_offline.bat
```

啟動前會先執行 `--dry-run` 健康檢查，通過後才啟動 Gradio。

## 本機網址與 Port 備援

- 預設優先使用：`http://127.0.0.1:7860`
- 若 7860 已被占用，程式會自動切換到：`http://127.0.0.1:7861`
- 終端機會印出實際使用的 URL。
- `share` 固定為 `False`，不會產生 `gradio.live` URL。

## Ollama 故障處理

若 Ollama 未啟動、`gemma3:4b` 未安裝或回應失敗：

- 模型分類不中斷。
- 五類機率、validation threshold 判定與 Ground Truth 比對仍會顯示。
- 報告改用固定保守規則式說明。
- 不會呼叫雲端 LLM。

## 重新啟動 Demo

執行：

```bat
scripts\restart_demo_offline.bat
```

此腳本只會停止命令列中包含本專案 `app_full_image_multilabel_ollama_gradio.py` 的 Python 程序，不會任意終止其他 Python 程式。停止後會重新執行健康檢查並啟動 Demo。

## 離線驗證模式

可執行：

```bat
python app_full_image_multilabel_ollama_gradio.py --project-root . --offline-smoke-test
```

驗證內容包含：模型推論、Ground Truth 查詢、Ollama 可用或不可用時的備援、`share=False`、不建立 `gradio.live`、以及 7860 被占用時切換 7861 的 port fallback。

## LAN 模式

預設不啟用 LAN。若比賽展示需要同網段其他裝置連線，可手動加入：

```bat
python app_full_image_multilabel_ollama_gradio.py --project-root . --server-port 7860 --lan
```

LAN 模式會使用 `server_name=0.0.0.0`，但 `share` 仍固定為 `False`。請只在可信任區網使用，並避免上傳含病人個資的影像或表單資訊。

## 比賽前檢查清單

- checkpoint 存在且 SHA256 符合正式值。
- validation thresholds 存在且 SHA256 符合正式值。
- Ground Truth catalog 可讀且列數正確。
- `scripts\start_demo_offline.bat` 可啟動。
- 7860 或 7861 至少一個可用。
- Ollama `gemma3:4b` 可用；若不可用，rule-based fallback 可正常顯示。
- 上傳 PNG/JPG 後模型可完成分類。
- 報告可列印或匯出 PDF。
- 沒有 `gradio.live` 公開網址。

## 主電腦故障備份方案

1. 準備另一台已安裝 Python、PyTorch、Gradio、Ollama 的 Windows 電腦。
2. 將本專案完整資料夾複製到備用電腦，確認 checkpoint、threshold、Ground Truth catalog 與 Demo 圖片存在。
3. 在備用電腦執行 `scripts\start_demo_offline.bat`。
4. 若 Ollama 無法及時恢復，Demo 仍可使用 rule-based 說明完成分類展示。
