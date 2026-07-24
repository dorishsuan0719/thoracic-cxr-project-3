# 10 Conflicts And Unknowns

建立時間：2026-07-23T04:33:24.534294+00:00

## 已發現衝突/風險來源（5）

- **舊版/輔助文件含 mojibake 中文**：`docs/final_patch_distillation_project/*.md` 部分文字為問號/亂碼；不可作正式敘述主要來源。
- **多個 comparison 資料夾範圍不同**：`final_comparison_report.md` 是 Proposed vs Baseline；正式三模型結論應優先用 `raddino_convnext_tiny_three_model_comparison_seed42`。
- **舊路徑引用出現在 path audit JSON**：`old_path_reference_scan*.json` 包含舊專案路徑，屬 audit 記錄，不代表正式程式依賴。
- **469 brightness generator 未找到**：factor 分布可由 manifest/filename 驗證，但產生器實作位置為 `〔待確認〕`。
- **指標名稱不可混用**：ROI Accuracy/macro-F1 與 Full-image macro-F1/micro-F1/exact subset accuracy 是不同任務指標。

## 待使用者確認項目（4）

- brightness augmented ROI 的原始生成腳本與抽樣策略：〔待確認〕。
- 原始 BBox 人工修正/審查決策若需逐筆解釋，需另讀人工 review 檔：〔待確認〕。
- 外部測試集泛化能力：目前證據主要為 seed 42 與 project-3 內部 split：〔待確認〕。
- Full-image multilabel 尚未見對應 bootstrap CI：〔待確認〕。
