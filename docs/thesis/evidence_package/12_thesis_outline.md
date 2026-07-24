# 12 Thesis Outline

## 第一章 緒論
- 研究背景：胸腔 X 光多疾病辨識與 AI 輔助展示系統。
- 問題定義：ROI 單標籤分類與 full-image 多標籤推論是不同任務。
- 研究限制：本系統非正式診斷；Ollama 不做影像判讀。

## 第二章 文獻與方法背景
- 胸腔 X 光分類與多標籤學習。
- Knowledge distillation / feature distillation。
- RAD-DINO 作為 frozen teacher；ConvNeXt-Tiny 作為 student/backbone。
- 文獻作者、年份、DOI 需另行查證，不可由本證據包捏造。

## 第三章 資料集建立與前處理
- Raw CXR、BBox annotation、ROI crop、ROI 224、資料稽核。
- Balanced ROI 4,725 張，每類 945；469 brightness augmented ROI。
- Full-image multilabel 590 張，train/val/test 472/59/59。

## 第四章 模型設計與訓練流程
- RAD-DINO CLS cache 與 patch cache。
- CLS distillation 與 Patch distillation。
- ROI Phase 2 三模型比較。
- Full-image Patch-transfer multilabel fine-tuning：BCEWithLogitsLoss + Sigmoid + validation threshold tuning。

## 第五章 實驗結果與分析
- ROI overall/per-class metrics 與 bootstrap CI。
- Full-image multilabel test macro/micro F1 與 per-label 指標。
- 不混用 ROI Accuracy 與 Full-image exact subset accuracy。

## 第六章 Demo 系統與臨床輔助說明
- Gradio UI、inference service、Ground Truth comparison、Ollama explanation、session/PDF output。
- GT 不輸入模型、不送 Ollama。
- 安全語氣與 disclaimer。

## 第七章 結論與未來工作
- 蒸餾模型尚未形成整體統計顯著優勢；Patch 對特定類別有 point improvement。
- Full-image 多標籤 Demo 已完成整合，但仍需外部驗證、更多 seed 與更嚴謹醫學評估。
