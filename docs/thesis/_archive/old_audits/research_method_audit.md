# 第三章研究方法逐句數據稽核

本報告只稽核 `docs/thesis/draft/胸腔X光影像分類與知識蒸餾系統_論文保守初稿.docx` 的第三章內容，未修改 DOCX。查證來源限於使用者指定之 evidence package 與 thesis evidence trace。

## 稽核摘要

| 項目 | 結果 |
|---|---|
| 第三章檢查區塊數 | 47 |
| 方法相關數字數 | 101 |
| 數字正確 | 100 |
| 數字需要修正 | 0 |
| 數字證據不足 | 1 |
| ROI/Full-image 混淆 | 否 |
| Activation/Loss 混淆 | 否 |
| YOLO/舊專案正式方法混入 | 否 |

## 特別檢查結論

| 檢查項目 | 結論 |
|---|---|
| Full image、BBox、ROI 三種資料層級 | 通過。第三章將 ROI 裁切流程與 Full-image 完整影像多標籤流程分開描述。 |
| ROI 單標籤分類與 Full-image 多標籤分類 | 通過。ROI 使用單標籤五分類；Full-image 使用五類多標籤。 |
| Softmax 與 Sigmoid | 通過。ROI 段落描述 Softmax 評估；Full-image 段落描述 Sigmoid 推論。 |
| Phase 0、Phase 1、Phase 2 | 通過。Phase 0 為 teacher feature cache，Phase 1 為 distillation，Phase 2 為 ROI classification 或 Full-image fine-tuning。 |
| RAD-DINO Teacher 與最終 ConvNeXt-Tiny | 通過。RAD-DINO 被描述為 frozen teacher，不是最終 Demo 推論模型。 |
| ROI Patch checkpoint 到 Full-image | 通過。第三章描述 ROI Patch Proposed 匯出 checkpoint 作為 Full-image 初始化來源，且丟棄 ROI head。 |
| 訓練設定與測試結果 | 需文字微調。3.8/3.9 與表 5 含 distillation loss/cosine 結果值，建議移至第四章或改稱訓練監控摘要。 |
| Validation 與 Test | 通過。Threshold 明確寫為 validation-selected，且 test_used 為 False。 |
| 469 張 brightness augmentation | 通過。第三章僅寫 factor 0.95 至 1.05，未宣稱 random.uniform 或均勻抽樣。 |
| YOLO 舊專案 | 通過。第三章未將 YOLO 舊專案寫入正式研究方法。 |

## 技術數字稽核清單

| Section | Item | Current value | Verified value | Source | Status |
|---|---|---|---|---|---|
| 3.1 | class_id_Aortic enlargement | 0 | 0 | `15_final_verified_thesis_facts.md` / 正式五類順序 | correct |
| 3.1 | class_id_Cardiomegaly | 1 | 1 | `15_final_verified_thesis_facts.md` / 正式五類順序 | correct |
| 3.1 | class_id_Pleural thickening | 2 | 2 | `15_final_verified_thesis_facts.md` / 正式五類順序 | correct |
| 3.1 | class_id_Pulmonary fibrosis | 3 | 3 | `15_final_verified_thesis_facts.md` / 正式五類順序 | correct |
| 3.1 | class_id_Pleural effusion | 4 | 4 | `15_final_verified_thesis_facts.md` / 正式五類順序 | correct |
| 3.4 | ROI original rows | 4546 | 4546 | `05_verified_dataset_counts.csv` / ROI 224 original/all/ALL/roi_rows | correct |
| 3.4 | ROI balanced rows | 4725 | 4725 | `05_verified_dataset_counts.csv` / ROI balanced feature cache/all/ALL/roi_rows | correct |
| 3.4 | ROI balanced per class | 945 | 945 | `05_verified_dataset_counts.csv` / ROI balanced feature cache/all/class/roi_rows | correct |
| 3.4 | ROI original rows kept in balanced cache | 4256 | 4256 | `05_verified_dataset_counts.csv` / original_roi_rows | correct |
| 3.4 | brightness augmented rows | 469 | 469 | `05_verified_dataset_counts.csv; 14_conflict_resolution.md` / brightness_augmented_rows | correct |
| 3.4 | brightness_factor_0.95 | 0.95 | 0.95 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.95_count | 22 | 22 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.96 | 0.96 | 0.96 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.96_count | 54 | 54 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.97 | 0.97 | 0.97 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.97_count | 73 | 73 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.98 | 0.98 | 0.98 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.98_count | 57 | 57 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.99 | 0.99 | 0.99 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_0.99_count | 26 | 26 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.01 | 1.01 | 1.01 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.01_count | 21 | 21 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.02 | 1.02 | 1.02 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.02_count | 53 | 53 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.03 | 1.03 | 1.03 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.03_count | 63 | 63 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.04 | 1.04 | 1.04 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.04_count | 75 | 75 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.05 | 1.05 | 1.05 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.4 | brightness_factor_1.05_count | 25 | 25 | `14_conflict_resolution.md` / Brightness factor table | correct |
| 3.5 | ROI split train total | 3770 | 3770 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/train/ALL/roi_rows | correct |
| 3.5 | ROI split train brightness | 357 | 357 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/train/ALL/brightness_augmented_rows | correct |
| 3.5 | ROI split train Aortic enlargement | 744 | 744 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/train/Aortic enlargement/roi_rows | correct |
| 3.5 | ROI split train Cardiomegaly | 759 | 759 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/train/Cardiomegaly/roi_rows | correct |
| 3.5 | ROI split train Pleural thickening | 763 | 763 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/train/Pleural thickening/roi_rows | correct |
| 3.5 | ROI split train Pulmonary fibrosis | 756 | 756 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/train/Pulmonary fibrosis/roi_rows | correct |
| 3.5 | ROI split train Pleural effusion | 748 | 748 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/train/Pleural effusion/roi_rows | correct |
| 3.5 | ROI split val total | 454 | 454 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/val/ALL/roi_rows | correct |
| 3.5 | ROI split val brightness | 0 | 0 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/val/ALL/brightness_augmented_rows | correct |
| 3.5 | ROI split val Aortic enlargement | 77 | 77 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/val/Aortic enlargement/roi_rows | correct |
| 3.5 | ROI split val Cardiomegaly | 78 | 78 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/val/Cardiomegaly/roi_rows | correct |
| 3.5 | ROI split val Pleural thickening | 112 | 112 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/val/Pleural thickening/roi_rows | correct |
| 3.5 | ROI split val Pulmonary fibrosis | 106 | 106 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/val/Pulmonary fibrosis/roi_rows | correct |
| 3.5 | ROI split val Pleural effusion | 81 | 81 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/val/Pleural effusion/roi_rows | correct |
| 3.5 | ROI split test total | 454 | 454 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/test/ALL/roi_rows | correct |
| 3.5 | ROI split test brightness | 0 | 0 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/test/ALL/brightness_augmented_rows | correct |
| 3.5 | ROI split test Aortic enlargement | 77 | 77 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/test/Aortic enlargement/roi_rows | correct |
| 3.5 | ROI split test Cardiomegaly | 78 | 78 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/test/Cardiomegaly/roi_rows | correct |
| 3.5 | ROI split test Pleural thickening | 112 | 112 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/test/Pleural thickening/roi_rows | correct |
| 3.5 | ROI split test Pulmonary fibrosis | 106 | 106 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/test/Pulmonary fibrosis/roi_rows | correct |
| 3.5 | ROI split test Pleural effusion | 81 | 81 | `05_verified_dataset_counts.csv` / ROI phase2 grouped split/test/Pleural effusion/roi_rows | correct |
| 3.6-3.10 | CLS cache samples | 4725 | 4725 | `07_model_training_settings.csv` / CLS cache samples | correct |
| 3.6-3.10 | CLS feature dimension | 768 | 768 | `07_model_training_settings.csv` / CLS feature dimension | correct |
| 3.6-3.10 | RAD-DINO patch processor channels | 3 | 3 | `07_model_training_settings.csv` / RAD-DINO patch processor channels | correct |
| 3.6-3.10 | RAD-DINO patch processor height | 518 | 518 | `07_model_training_settings.csv` / RAD-DINO patch processor height | correct |
| 3.6-3.10 | RAD-DINO patch processor width | 518 | 518 | `07_model_training_settings.csv` / RAD-DINO patch processor width | correct |
| 3.6-3.10 | Patch cache samples | 4725 | 4725 | `07_model_training_settings.csv` / Patch cache samples | correct |
| 3.6-3.10 | Patch feature channels | 768 | 768 | `07_model_training_settings.csv` / Patch feature channels | correct |
| 3.6-3.10 | Patch feature grid height | 7 | 7 | `07_model_training_settings.csv` / Patch feature grid height | correct |
| 3.6-3.10 | Patch feature grid width | 7 | 7 | `07_model_training_settings.csv` / Patch feature grid width | correct |
| 3.6-3.10 | CLS distillation completed epochs | 30 | 30 | `07_model_training_settings.csv` / CLS distillation completed epochs | correct |
| 3.6-3.10 | CLS distillation effective batch size | 64 | 64 | `07_model_training_settings.csv` / CLS distillation effective batch size | correct |
| 3.6-3.10 | Patch distillation completed epochs | 84 | 84 | `07_model_training_settings.csv` / Patch distillation completed epochs | correct |
| 3.6-3.10 | Patch distillation effective batch size | 64 | 64 | `07_model_training_settings.csv` / Patch distillation effective batch size | correct |
| 3.6-3.10 | ROI classifier logits | 5 | 5 | `07_model_training_settings.csv` / ROI classifier logits | correct |
| 3.6-3.10 | ROI backbone learning rate | 1e-05 | 1e-05 | `07_model_training_settings.csv` / ROI backbone learning rate | correct |
| 3.6-3.10 | ROI head learning rate | 0.0001 | 0.0001 | `07_model_training_settings.csv` / ROI head learning rate | correct |
| 3.6-3.10 | ROI effective batch size | 64 | 64 | `07_model_training_settings.csv` / ROI effective batch size | correct |
| 3.6-3.10 | ROI maximum epochs | 50 | 50 | `07_model_training_settings.csv` / ROI maximum epochs | correct |
| 3.6-3.10 | ROI early stopping patience | 10 | 10 | `07_model_training_settings.csv` / ROI early stopping patience | correct |
| 3.13 | Full-image train images | 472 | 472 | `05_verified_dataset_counts.csv` / Full-image phase0 split/train/ALL/image_rows | correct |
| 3.13 | Full-image train Aortic enlargement positives | 280 | 280 | `05_verified_dataset_counts.csv` / Full-image phase0 split/train/Aortic enlargement/positive_images | correct |
| 3.13 | Full-image train Cardiomegaly positives | 279 | 279 | `05_verified_dataset_counts.csv` / Full-image phase0 split/train/Cardiomegaly/positive_images | correct |
| 3.13 | Full-image train Pleural thickening positives | 280 | 280 | `05_verified_dataset_counts.csv` / Full-image phase0 split/train/Pleural thickening/positive_images | correct |
| 3.13 | Full-image train Pulmonary fibrosis positives | 278 | 278 | `05_verified_dataset_counts.csv` / Full-image phase0 split/train/Pulmonary fibrosis/positive_images | correct |
| 3.13 | Full-image train Pleural effusion positives | 280 | 280 | `05_verified_dataset_counts.csv` / Full-image phase0 split/train/Pleural effusion/positive_images | correct |
| 3.13 | Full-image val images | 59 | 59 | `05_verified_dataset_counts.csv` / Full-image phase0 split/val/ALL/image_rows | correct |
| 3.13 | Full-image val Aortic enlargement positives | 35 | 35 | `05_verified_dataset_counts.csv` / Full-image phase0 split/val/Aortic enlargement/positive_images | correct |
| 3.13 | Full-image val Cardiomegaly positives | 36 | 36 | `05_verified_dataset_counts.csv` / Full-image phase0 split/val/Cardiomegaly/positive_images | correct |
| 3.13 | Full-image val Pleural thickening positives | 35 | 35 | `05_verified_dataset_counts.csv` / Full-image phase0 split/val/Pleural thickening/positive_images | correct |
| 3.13 | Full-image val Pulmonary fibrosis positives | 36 | 36 | `05_verified_dataset_counts.csv` / Full-image phase0 split/val/Pulmonary fibrosis/positive_images | correct |
| 3.13 | Full-image val Pleural effusion positives | 35 | 35 | `05_verified_dataset_counts.csv` / Full-image phase0 split/val/Pleural effusion/positive_images | correct |
| 3.13 | Full-image test images | 59 | 59 | `05_verified_dataset_counts.csv` / Full-image phase0 split/test/ALL/image_rows | correct |
| 3.13 | Full-image test Aortic enlargement positives | 35 | 35 | `05_verified_dataset_counts.csv` / Full-image phase0 split/test/Aortic enlargement/positive_images | correct |
| 3.13 | Full-image test Cardiomegaly positives | 35 | 35 | `05_verified_dataset_counts.csv` / Full-image phase0 split/test/Cardiomegaly/positive_images | correct |
| 3.13 | Full-image test Pleural thickening positives | 35 | 35 | `05_verified_dataset_counts.csv` / Full-image phase0 split/test/Pleural thickening/positive_images | correct |
| 3.13 | Full-image test Pulmonary fibrosis positives | 36 | 36 | `05_verified_dataset_counts.csv` / Full-image phase0 split/test/Pulmonary fibrosis/positive_images | correct |
| 3.13 | Full-image test Pleural effusion positives | 35 | 35 | `05_verified_dataset_counts.csv` / Full-image phase0 split/test/Pleural effusion/positive_images | correct |
| 3.13-3.14 | Full-image input channels | 3 | 3 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Full-image input channels | correct |
| 3.13-3.14 | Full-image input height | 224 | 224 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Full-image input height | correct |
| 3.13-3.14 | Full-image input width | 224 | 224 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Full-image input width | correct |
| 3.13-3.14 | Full-image logits | 5 | 5 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Full-image logits | correct |
| 3.13-3.14 | Threshold search minimum | 0.05 | 0.05 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Threshold search minimum | correct |
| 3.13-3.14 | Threshold search maximum | 0.95 | 0.95 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Threshold search maximum | correct |
| 3.13-3.14 | Threshold search step | 0.01 | 0.01 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Threshold search step | correct |
| 3.13-3.14 | Aortic enlargement threshold | 0.50 | 0.50 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Aortic enlargement threshold | correct |
| 3.13-3.14 | Cardiomegaly threshold | 0.50 | 0.50 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Cardiomegaly threshold | correct |
| 3.13-3.14 | Pleural thickening threshold | 0.39 | 0.39 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Pleural thickening threshold | correct |
| 3.13-3.14 | Pulmonary fibrosis threshold | 0.36 | 0.36 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Pulmonary fibrosis threshold | correct |
| 3.13-3.14 | Pleural effusion threshold | 0.34 | 0.34 | `07_model_training_settings.csv; validation_selected_thresholds.json` / Pleural effusion threshold | correct |
| 3.16 | 完整硬體與套件版本 | 〔待確認〕 | 〔待確認〕 | `16_unresolved_items_for_user.md` / 仍需使用者決定或補充資料 | insufficient |

## 逐段稽核

| # | Section | Type | Status | Note | Text excerpt |
|---|---|---|---|---|---|
| 1 | 第三章 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 第三章　研究方法 |
| 2 | 3.1 整體研究架構 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.1 整體研究架構 |
| 3 | 3.1 整體研究架構 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 整體流程可分為 ROI 階段與 Full-image 階段。ROI 階段先以 BBox 標註建立五類 ROI 影像，再建立 balanced ROI dataset，並進行 RAD-DINO CLS 與 patch feature cache。其後將 feature cache 作為 teacher 訊號訓練 ConvNeXt-Tiny student，最後進行 ROI 五類單標籤分類比較。Full-image 階段則以 ROI Patch Proposed checkpoint 匯出的 backbone 作為初始化，訓練完整胸腔 X 光五類多標籤模型。 |
| 4 | 3.1 整體研究架構 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 表 1　正式五類與任務定義 |
| 5 | 3.1 整體研究架構 | table | OK | 已比對 evidence；未發現數據錯置。 | 類別編號 / 英文類別名稱 / ROI 階段 / Full-image 階段 / 0 / Aortic enlargement / 單標籤分類候選類別 / 多標籤獨立陽性標籤 / 1 / Cardiomegaly / 單標籤分類候選類別 / 多標籤獨立陽性標籤 / 2 / Pleural thickening / 單標籤分類候選類別 / 多標籤獨立陽性標籤 / 3 / Pulmonary fibrosis / 單標籤分類候選類別 / 多標籤獨立陽性標籤 / 4 / Pleural effusion / 單標籤分類候選類別 / 多標籤獨立陽性標籤 |
| 6 | 3.2 資料來源與五類疾病 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.2 資料來源與五類疾病 |
| 7 | 3.2 資料來源與五類疾病 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 本研究的正式五類為 Aortic enlargement、Cardiomegaly、Pleural thickening、Pulmonary fibrosis 與 Pleural effusion，不包含 No Finding 類別。ROI 階段使用經 BBox 裁切後的單標籤資料；Full-image 階段則使用完整胸腔 X 光影像與五類多標籤 ground truth。 |
| 8 | 3.3 BBox 標註與 ROI 製作 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.3 BBox 標註與 ROI 製作 |
| 9 | 3.3 BBox 標註與 ROI 製作 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | ROI 製作流程依據 BBox 標註裁切病灶區域，並產生 224×224 ROI 版本供 feature cache、distillation 與分類訓練使用。現有 evidence 可確認曾進行 BBox overlay、crop 與 resize summary 層級人工檢查；逐筆人工審查紀錄目前未納入證據包，因此本文僅保守描述為 summary 層級品質檢查。 |
| 10 | 3.4 ROI 平衡與資料增強 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.4 ROI 平衡與資料增強 |
| 11 | 3.4 ROI 平衡與資料增強 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | ROI 原始資料共 4546 筆 ROI rows；balanced feature cache 共 4725 筆，五類各 945 筆。其中 4256 筆為原始 ROI，469 筆為可由 manifest 與檔名反向驗證之 brightness augmentation 影像。由於目前未找到 brightness generator 程式，本文不宣稱其產生方式為特定隨機公式。 |
| 12 | 3.4 ROI 平衡與資料增強 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 表 2　ROI balanced dataset 類別數量 |
| 13 | 3.4 ROI 平衡與資料增強 | table | OK | 已比對 evidence；未發現數據錯置。 | 類別 / ROI rows / Brightness augmented rows / Aortic enlargement / 945 / 173 / Cardiomegaly / 945 / 162 / Pleural thickening / 945 / 0 / Pulmonary fibrosis / 945 / 0 / Pleural effusion / 945 / 134 |
| 14 | 3.4 ROI 平衡與資料增強 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 表 3　Brightness augmentation factor 反向驗證分布 |
| 15 | 3.4 ROI 平衡與資料增強 | table | OK | 已比對 evidence；未發現數據錯置。 | Factor / 影像數量 / 0.95 / 22 / 0.96 / 54 / 0.97 / 73 / 0.98 / 57 / 0.99 / 26 / 1.01 / 21 / 1.02 / 53 / 1.03 / 63 / 1.04 / 75 / 1.05 / 25 |
| 16 | 3.5 Source-level 資料切分及資料洩漏控制 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.5 Source-level 資料切分及資料洩漏控制 |
| 17 | 3.5 Source-level 資料切分及資料洩漏控制 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 表 4　ROI Phase2 grouped split 類別數量 |
| 18 | 3.5 Source-level 資料切分及資料洩漏控制 | table | OK | 已比對 evidence；未發現數據錯置。 | Split / Total / Brightness rows / Aortic enlargement / Cardiomegaly / Pleural thickening / Pulmonary fibrosis / Pleural effusion / train / 3770 / 357 / 744 / 759 / 763 / 756 / 748 / val / 454 / 0 / 77 / 78 / 112 / 106 / 81 / test / 454 / 0 / 77 / 78 / 112 / 106 / 81 |
| 19 | 3.5 Source-level 資料切分及資料洩漏控制 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | ROI Phase2 grouped split 中，train split 含 brightness augmentation；validation 與 test split 的 brightness augmented rows 皆為 0。此設定可避免以增強影像污染 validation 或 test 評估。 |
| 20 | 3.6 Phase 0：RAD-DINO CLS Feature Cache | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.6 Phase 0：RAD-DINO CLS Feature Cache |
| 21 | 3.6 Phase 0：RAD-DINO CLS Feature Cache | paragraph | OK | 已比對 evidence；未發現數據錯置。 | RAD-DINO CLS teacher cache 使用 frozen/eval 模式，針對 ROI 224 balanced manifest 建立 teacher features，輸出形狀為 [4725, 768]。此階段僅進行 teacher feature extraction，不訓練 teacher。 |
| 22 | 3.7 Phase 0：RAD-DINO Patch Feature Cache | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.7 Phase 0：RAD-DINO Patch Feature Cache |
| 23 | 3.7 Phase 0：RAD-DINO Patch Feature Cache | paragraph | OK | 已比對 evidence；未發現數據錯置。 | RAD-DINO Patch teacher cache 同樣使用 frozen/eval 模式。其處理器輸入為 [B, 3, 518, 518]，輸出 patch teacher features 形狀為 [4725, 768, 7, 7]，保留 7×7 空間格點表徵。 |
| 24 | 3.8 Phase 1：CLS Feature Distillation | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.8 Phase 1：CLS Feature Distillation |
| 25 | 3.8 Phase 1：CLS Feature Distillation | paragraph | OK | 已比對 evidence；未發現數據錯置。 | CLS distillation 以 ConvNeXt-Tiny student 學習 RAD-DINO CLS feature。Loss 為 L2-normalized student feature 與 L2-normalized teacher feature 之 MSE；optimizer 為 AdamW；有效 batch size 為 64；訓練完成 30 epochs，checkpoint selection 依 best average distillation loss。 |
| 26 | 3.9 Phase 1：Patch Feature Distillation | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.9 Phase 1：Patch Feature Distillation |
| 27 | 3.9 Phase 1：Patch Feature Distillation | paragraph | OK | 已比對 evidence；未發現數據錯置。 | Patch distillation 以 ConvNeXt-Tiny ImageNet1K V1 student 學習 RAD-DINO 7×7 patch feature。Loss 為沿 channel dimension 進行 L2 normalization 後的 float32 MSE；有效 batch size 為 64；訓練完成 84 epochs，並以 patch MSE/cosine 監控選擇最佳 checkpoint。 |
| 28 | 3.9 Phase 1：Patch Feature Distillation | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 表 5　Phase 1 distillation 摘要 |
| 29 | 3.9 Phase 1：Patch Feature Distillation | table | WORDING_REVIEW | 數字可查證，但屬訓練結果/監控值，建議移至結果章或改稱訓練監控摘要。 | 階段 / Completed epochs / Checkpoint selection / MSE / loss / Cosine similarity / CLS distillation / 30 / Best average distillation loss / 0.0003958256 / 0.8480030057 / Patch distillation / 84 / Best monitor patch MSE/cosine / 0.0002644237 / 0.89846133 |
| 30 | 3.10 Phase 2：ROI 五分類模型 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.10 Phase 2：ROI 五分類模型 |
| 31 | 3.10 Phase 2：ROI 五分類模型 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | ROI Phase2 為五類單標籤分類，輸出 5 logits，評估時使用 Softmax 機率。三個模型分別為 ImageNet Baseline、RAD-DINO CLS Proposed 與 RAD-DINO Patch Proposed。三者皆使用 CrossEntropyLoss、AdamW with CosineAnnealingLR、backbone learning rate 1e-05、head learning rate 0.0001、有效 batch size 64、最大 50 epochs，checkpoint selection 以 best validation l |
| 32 | 3.11 ROI 三模型比較設計 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.11 ROI 三模型比較設計 |
| 33 | 3.11 ROI 三模型比較設計 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 正式 ROI 比較採三模型公平比較資料夾中的輸出，不採早期二模型比較。公平性稽核顯示 augmentation 設定鎖定且 train-only，因此三模型比較具一致實驗條件。 |
| 34 | 3.12 ROI Patch Backbone 轉移 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.12 ROI Patch Backbone 轉移 |
| 35 | 3.12 ROI Patch Backbone 轉移 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | Full-image 階段的初始化來源為 ROI RAD-DINO Patch Proposed 匯出的 ConvNeXt-Tiny 五類 checkpoint。轉移時丟棄 ROI head，改接 Full-image 多標籤任務的新分類 head。此作法將 ROI patch-level distillation 學到的 backbone 表徵帶入完整影像任務。 |
| 36 | 3.13 Full-image 多標籤 Fine-tuning | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.13 Full-image 多標籤 Fine-tuning |
| 37 | 3.13 Full-image 多標籤 Fine-tuning | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 表 6　Full-image 多標籤 split 與每類 positive count |
| 38 | 3.13 Full-image 多標籤 Fine-tuning | table | OK | 已比對 evidence；未發現數據錯置。 | Split / Images / Aortic enlargement / Cardiomegaly / Pleural thickening / Pulmonary fibrosis / Pleural effusion / train / 472 / 280 / 279 / 280 / 278 / 280 / val / 59 / 35 / 36 / 35 / 36 / 35 / test / 59 / 35 / 35 / 35 / 36 / 35 |
| 39 | 3.13 Full-image 多標籤 Fine-tuning | paragraph | OK | 已比對 evidence；未發現數據錯置。 | Full-image 階段使用完整胸腔 X 光影像，不使用 BBox、ROI crop 或資料增強。輸入影像 resize 為 [3,224,224]；模型輸出五個 raw logits，訓練 loss 為 BCEWithLogitsLoss，推論時使用 Sigmoid 取得五類獨立機率。 |
| 40 | 3.14 Validation Threshold | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.14 Validation Threshold |
| 41 | 3.14 Validation Threshold | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 表 7　Validation-tuned thresholds |
| 42 | 3.14 Validation Threshold | table | OK | 已比對 evidence；未發現數據錯置。 | 類別 / Threshold / Validation F1 / Tie break / Aortic enlargement / 0.5 / 0.8493 / closest_to_0.5_then_lower_threshold / Cardiomegaly / 0.5 / 0.9333 / closest_to_0.5_then_lower_threshold / Pleural thickening / 0.39 / 0.8767 / closest_to_0.5_then_lower_threshold / Pulmonary fibrosis / 0.36 / 0.8000 / c |
| 43 | 3.14 Validation Threshold | paragraph | OK | 已比對 evidence；未發現數據錯置。 | Threshold 搜尋範圍為 0.05 至 0.95，step 為 0.01；selection metric 為 per-class Validation F1。`validation_selected_thresholds.json` 顯示 `test_used` 為 False，因此 test set 未用於 threshold 選擇。 |
| 44 | 3.15 評估指標 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.15 評估指標 |
| 45 | 3.15 評估指標 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | ROI 單標籤分類使用 Accuracy、Macro-F1、Weighted-F1、Macro-AUROC 等指標。Full-image 多標籤分類使用 Macro-F1、Micro-F1、Exact Subset Accuracy、Hamming Loss、Macro-AUROC、Micro-AUROC 與各類 precision、recall、F1。Exact Subset Accuracy 表示一張影像所有標籤完全一致的比例，不能與 ROI 單標籤 Accuracy 直接等同。 |
| 46 | 3.16 實驗環境 | paragraph | OK | 已比對 evidence；未發現數據錯置。 | 3.16 實驗環境 |
| 47 | 3.16 實驗環境 | paragraph | INSUFFICIENT_EVIDENCE_MARKED | 原文已保留待確認，未硬寫成結論。 | 本研究之正式訓練程式與輸出紀錄可追蹤至 evidence package；然而硬體型號、作業系統版本、CUDA 與 PyTorch 版本等完整環境資訊目前未在第二輪證據中統一整理，因此本小節保留為〔待確認〕。 |

## 建議修正

詳見 `docs/thesis/draft/research_method_corrections.csv`。本輪主要屬文字位置與證據不足標記，未發現錯誤數字。
