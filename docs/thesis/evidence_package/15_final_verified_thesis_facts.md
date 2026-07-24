# 15 Final Verified Thesis Facts

本文件只保留第二輪查證後可直接寫入論文的正式事實。每項均附來源；未能由專案證據確認的內容不列入本文件。

## 1. 正式研究任務

1. ROI 階段為五類單標籤分類，使用 Softmax 類型分類設定。
   - 來源：`AGENTS.md`；`outputs/raddino_convnext_tiny_three_model_comparison_seed42/tables/overall_metrics_comparison.csv`；Phase2 ROI training scripts。

2. Full-image 階段為五類多標籤分類，使用 Sigmoid 與 multilabel threshold。
   - 來源：`src/train_full_image_224_multilabel_patch_transfer.py` 中 `BCEWithLogitsLoss` 與 `torch.sigmoid`；`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/test_metrics.json`。

3. 正式五類為 Aortic enlargement、Cardiomegaly、Pleural thickening、Pulmonary fibrosis、Pleural effusion。
   - 來源：`AGENTS.md`；`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/per_class_test_metrics.csv`。

4. 最終 Demo 使用 Full-image ConvNeXt-Tiny multilabel 模型，不使用 YOLO、BBox、ROI crop 或 Softmax 單標籤推論流程。
   - 來源：`app_full_image_multilabel_ollama_gradio.py`；`src/full_image_multilabel_inference_service.py`。

## 2. ROI 資料集與 brightness augmentation

1. Balanced ROI dataset 共 4,725 張，五類各 945 張。
   - 來源：`outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv`。

2. Balanced ROI manifest 中含 469 張 brightness augmentation 影像。
   - 來源：`outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv` 欄位與檔名解析。

3. Brightness factor 實際出現值與數量如下：

| factor | count |
|---:|---:|
| 0.95 | 22 |
| 0.96 | 54 |
| 0.97 | 73 |
| 0.98 | 57 |
| 0.99 | 26 |
| 1.01 | 21 |
| 1.02 | 53 |
| 1.03 | 63 |
| 1.04 | 75 |
| 1.05 | 25 |

4. Phase2 ROI train split 中有 357 張 brightness augmentation；val/test split 中無 brightness augmentation。
   - 來源：`outputs/raddino_convnext_tiny_experiment_seed42/phase2_split/train_roi_manifest.csv`、`val_roi_manifest.csv`、`test_roi_manifest.csv`。

5. 論文可描述 brightness augmentation 為「由輸出 manifest 與檔名反向驗證到的離線補充影像」；不可描述其產生器使用 `random.uniform(0.95, 1.05)`，除非後續找到產生器證據。
   - 來源：`src/audit_balanced_roi_and_build_manifest.py` 僅提供解析與稽核；目前未找到 generator。

## 3. Gaussian blur 與 Gaussian noise augmentation

1. Phase1 distillation 與 Phase1 patch distillation 使用 Gaussian blur 與 Gaussian noise 作為 train-time online augmentation。
   - 來源：`src/train_convnext_tiny_phase1_distillation.py` 的 `StudentTransform` 與 `augmentation_config`；`src/train_convnext_tiny_phase1_patch_distillation.py` 的 `StudentTransform` 與 `augmentation_config`；對應 config JSON。

2. Phase2 ROI fine-tuning 使用 Gaussian blur 與 Gaussian noise，設定為 train-only online augmentation。
   - 來源：`src/train_phase2_convnext_tiny_finetune.py` 的 `Phase2Transform` 與 `augmentation_config`；`outputs/raddino_convnext_tiny_patch_experiment_seed42/phase2_proposed_patch_distilled/config/shared_phase2_finetune_config.json`。

3. 正式三模型 ROI 比較中的 augmentation 設定已鎖定，公平性稽核顯示 train-only。
   - 來源：`outputs/raddino_convnext_tiny_three_model_comparison_seed42/fairness_audit.json`。

4. Full-image multilabel patch transfer 正式 training config 中 augmentation 為 false。
   - 來源：`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/training_config.json`；`preprocessing_spec.json`。

## 4. RAD-DINO 與 ConvNeXt-Tiny

1. RAD-DINO 在本專題中作為 teacher 或 feature cache 來源，不是最終 Demo 推論模型。
   - 來源：`AGENTS.md`；Phase1 distillation scripts；`outputs/raddino_feature_cache/`。

2. ConvNeXt-Tiny 為 ROI distillation 與 Full-image Demo 的主要 student/backbone 架構。
   - 來源：Phase2 ROI checkpoints；`src/full_image_multilabel_inference_service.py`。

## 5. 正式 checkpoint

1. ROI ImageNet Baseline checkpoint：
   - `outputs/raddino_convnext_tiny_experiment_seed42/phase2_baseline_imagenet/checkpoints/best.pt`
   - SHA256：`79311cb38d314d85aff6d2e1a9bc91e4f10916346425de7086dde33b2bfc74ff`

2. ROI RAD-DINO CLS Proposed checkpoint：
   - `outputs/raddino_convnext_tiny_experiment_seed42/phase2_proposed_distilled/checkpoints/best.pt`
   - SHA256：`6b0f8770d28f35df1641768ac5141f67f7b9bce1108afdf55c3b71edc17afdac`

3. ROI RAD-DINO Patch Proposed checkpoint：
   - `outputs/raddino_convnext_tiny_patch_experiment_seed42/phase2_proposed_patch_distilled/checkpoints/best.pt`
   - SHA256：`174e9e80aacd52c777670fc5331b5a908959d86080db2d05170943c8f5f3d82e`

4. Full-image 初始化來源 checkpoint：
   - `outputs/raddino_convnext_tiny_patch_experiment_seed42/phase2_proposed_patch_distilled/checkpoints/patch_proposed_convnext_tiny_5class.pt`
   - SHA256：`8a68d68b901d721c63a38b5e75ee3291a8c06d13195572d20f29fd34a56485e5`
   - 來源佐證：`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/initialization_audit.json`

5. Full-image training best checkpoint：
   - `outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/checkpoints/best.pt`
   - SHA256：`a1d4488dea57d9bdc034e36d59e414627c1efa21f07b52a95874c59c64693e0d`

6. Demo 實際部署 checkpoint：
   - `outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/checkpoints/full_image_multilabel_patch_transfer.pt`
   - SHA256：`0287fe36d3623ccdb5aa43857db1168a1598788071ebdecbc43324a6953f426f`
   - 來源佐證：`app_startup_audit.json`；`src/full_image_multilabel_inference_service.py`。

## 6. ROI 正式指標

正式來源：`outputs/raddino_convnext_tiny_three_model_comparison_seed42/tables/overall_metrics_comparison.csv`。

| model | Accuracy | Macro-F1 |
|---|---:|---:|
| ImageNet Baseline | 0.799559471366 | 0.807999579266 |
| RAD-DINO CLS Proposed | 0.792951541850 | 0.801748192622 |
| RAD-DINO Patch Proposed | 0.799559471366 | 0.805259370123 |

Bootstrap 結果顯示 principal metrics 的 95% CI 均跨越 0，因此不可宣稱 Patch Proposed 對 Baseline 具有統計顯著優勢。

- 來源：`outputs/raddino_convnext_tiny_three_model_comparison_seed42/statistics/cluster_bootstrap_results.csv`；`research_conclusion.md`。

## 7. Full-image 正式指標

正式來源：`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/test_metrics.json`。

| metric | value |
|---|---:|
| Macro-F1 | 0.7865085676876251 |
| Micro-F1 | 0.7858942065491183 |
| Exact Subset Accuracy | 0.1694915254237288 |
| Hamming Loss | 0.288135593220339 |
| Macro-AUROC | 0.7659989648033126 |
| Micro-AUROC | 0.7759740259740261 |

Validation thresholds 來源為 validation split，不使用 test split 選 threshold。

- 來源：`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/validation_selected_thresholds.json`。

## 8. Demo 與報告生成

1. Gradio Demo 進行圖片上傳、Full-image multilabel 推論、五類機率顯示、Ground Truth 比對、Ollama 說明與 PDF/HTML/Markdown 報告輸出。
   - 來源：`app_full_image_multilabel_ollama_gradio.py`。

2. Ground Truth lookup 只作 Demo 驗證與比對，不輸入模型，也不傳給 Ollama。
   - 來源：`src/full_image_multilabel_inference_service.py` 的 `_lookup_ground_truth` 與 `ollama_payload`；`app_full_image_multilabel_ollama_gradio.py` 中 `ground_truth_sent_to_ollama`。

3. Ollama 不負責影像分類，只根據模型輸出的結構化結果生成保守文字說明。
   - 來源：`src/full_image_multilabel_ollama_service.py`；`src/full_image_multilabel_report_prompt.py`。

## 9. 論文可安全採用的總體描述

「本研究先於 ROI 階段建立五類平衡資料集，使用 RAD-DINO 作為 teacher，將 CLS 與 patch-level feature distillation 至 ConvNeXt-Tiny，並與 ImageNet baseline 進行公平比較。雖然 Patch Proposed 在部分指標具競爭力，但 bootstrap 統計未支持其相對 baseline 的顯著優勢。後續將 ROI Patch Proposed backbone 轉移至 Full-image 五類多標籤任務，並整合 Gradio Demo、Ground Truth 比對、Ollama 輔助說明與報告匯出功能。」

