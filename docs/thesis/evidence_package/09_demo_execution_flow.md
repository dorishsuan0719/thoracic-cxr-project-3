# 09 Demo Execution Flow

建立時間：2026-07-23T04:33:24.534294+00:00

## 啟動與資源

- 主程式：`app_full_image_multilabel_ollama_gradio.py`
- UI layout version：`2026-07-22-exact-mockup-v1`
- Gradio version：`6.20.0`
- Model：`C:\Users\09688\thoracic-cxr-project-3\outputs\full_image_224_multilabel_seed42\phase2_patch_transfer\checkpoints\full_image_multilabel_patch_transfer.pt`
- Threshold：`C:\Users\09688\thoracic-cxr-project-3\outputs\full_image_224_multilabel_seed42\phase2_patch_transfer\validation_selected_thresholds.json`
- Ground Truth：`C:\Users\09688\thoracic-cxr-project-3\outputs\full_image_ground_truth_catalog\full_image_ground_truth_manifest.csv`
- Ollama：`http://127.0.0.1:11434`, model `gemma3:4b`

## Start Analysis Flow

1. `analyze_button.click()` 先更新排隊狀態。
2. `analyze_event(image_input, record_number, patient_name, patient_sex, patient_age, exam_date, patient_note)` 呼叫 `analyze_stream()`。
3. `FullImageMultilabelInferenceService.predict()` 讀單張圖，轉 RGB、resize 224、ImageNet normalization，ConvNeXt-Tiny 得到 logits `[1,5]`。
4. 使用 `torch.sigmoid` 得到五類獨立機率，套 validation thresholds `{'0': 0.5, '1': 0.5, '2': 0.39, '3': 0.36, '4': 0.34}` 得到 predicted vector。
5. Ground Truth lookup 在推論後依 image_id/sha/filename 查 catalog，僅用於 Demo 驗證。
6. `ollama_payload()` 建立要傳給 Ollama 的單張模型結果；不含 Ground Truth、不含整批 metrics、不含 confusion matrix。
7. Ollama 生成失敗時 fallback，分類結果保持不變。
8. `persist_session()` 輸出 prediction、Ground Truth comparison、session audit、Markdown 與 PDF。

## Callback 契約

- `analyze_event`：7 inputs、14 outputs、generator=True。
- `regenerate_event`：7 inputs、6 outputs。
- `clear_all`：0 inputs、20 outputs。
- 證據：`server_runtime_validation_report.json`, `callback_validation_report.json`。

## 安全隔離

- `run_smoke_test()` 檢查 prompt 不可含 `Ground Truth` 或 `ground_truth` key。
- `smoke_test_audit.json`：`ground_truth_sent_to_ollama=false`, `image_sent_to_ollama=false`。
- `callback_validation_report.json`：validation、GT miss、Ollama unavailable fallback 均產生 PDF/MD 並保留 disclaimer。
