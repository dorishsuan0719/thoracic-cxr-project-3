# Ollama 設定

## 固定連線

本 Demo 只允許本機 `http://127.0.0.1:11434`，健康檢查使用 `GET /api/tags`，生成使用 `POST /api/chat`、`stream=false`。程式不會連線 Ollama Cloud、OpenAI 相容端點或其他雲端服務。

確認服務與模型：

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

目前驗證模型為 `gemma3:4b`。本程式不會自動執行 `ollama pull`；需要其他本機模型時，請先自行安裝，再以 `--ollama-model` 明確指定。

## 模型選擇順序

1. CLI `--ollama-model`
2. 環境變數 `OLLAMA_MODEL`
3. 專案 config、`.env.example`、`llm_service.py`、`app.py` 或 README 中的明確設定
4. `/api/tags` 唯一可用本機模型

若有多個模型且無法唯一判斷，程式會停止並要求明確指定，不會猜測。

## Timeout 與離線行為

預設 timeout 為 120 秒，單次失敗最多重試一次。Ollama 離線、timeout 或輸出未通過規則時，分類結果維持顯示，中文說明區顯示本機服務錯誤。沒有雲端、Gemini 或其他 LLM fallback。

圖片與 base64 永遠不傳送給 Ollama；只傳送模型分類後的結構化文字與數值。
