# 11 Verified Thesis Facts

建立時間：2026-07-23T04:33:24.534294+00:00

## 資料與任務

- 正式五類：Aortic enlargement, Cardiomegaly, Pleural thickening, Pulmonary fibrosis, Pleural effusion；沒有 No Finding 類別。
- ROI 階段：五類單標籤分類，Softmax/CrossEntropyLoss。
- Full-image 階段：五類多標籤分類，Sigmoid/BCEWithLogitsLoss。
- ROI balanced dataset：4,725 張，每類 945；4,256 original ROI + 469 fixed brightness augmented ROI。
- Full-image multilabel dataset：590 張；train/val/test = 472/59/59；各類總 positive count 350。

## Feature / Distillation

- RAD-DINO CLS teacher cache：model `microsoft/rad-dino`, revision `110cbc18d5133582e320b43d53bf5c44e410c936`, feature shape `[4725, 768]`。
- RAD-DINO patch teacher cache：pooled patch feature shape `[4725, 768, 7, 7]`。
- Patch distillation：completed epochs 84，best monitor patch MSE `0.0002644237`，best monitor patch cosine `0.89846133`。

## ROI Classification Results

- ROI ImageNet Baseline：test accuracy 0.799559、macro-F1 0.808000。
- ROI RAD-DINO CLS Proposed：test accuracy 0.792952、macro-F1 0.801748。
- ROI RAD-DINO Patch Proposed：test accuracy 0.799559、macro-F1 0.805259。
- Pleural thickening F1 point estimates：Baseline 0.545455、CLS 0.541667、Patch 0.601852。
- Principal source-cluster bootstrap CIs all include zero；不可宣稱 Patch Proposed 全面或統計顯著勝過 Baseline。

## Full-image Results

- Full-image checkpoint 由 ROI Patch Proposed 初始化；舊 ROI head discarded，新 head 為 `Dropout(0.2) -> Linear(768,5)`。
- Full-image training：completed epochs 38，best epoch 28，best Validation Macro-AUROC 0.86420980。
- Full-image test：macro-F1 `0.786509`，micro-F1 `0.785894`，exact subset accuracy `0.169492`，hamming loss `0.288136`。
- Validation thresholds：{'0': 0.5, '1': 0.5, '2': 0.39, '3': 0.36, '4': 0.34}；`test_used=false`。

## Demo System

- Final Demo 使用 Full-image ConvNeXt-Tiny；不使用 YOLO、BBox、ROI crop 或 Softmax。
- Ground Truth catalog 590 rows；GT 在推論後查詢，不輸入模型，也不傳給 Ollama。
- Ollama `gemma3:4b` 只產生保守白話說明，不負責影像分類。
