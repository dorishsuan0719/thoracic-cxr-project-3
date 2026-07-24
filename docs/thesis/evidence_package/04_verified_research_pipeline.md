# 04 Verified Research Pipeline

建立時間：2026-07-23T04:33:24.534294+00:00

| Stage | 已驗證流程 | 主要程式 | 輸入 | 輸出/正式證據 | 論文可用事實 |
|---|---|---|---|---|---|
| 1 | Raw CXR 與 BBox annotation 整理 | `src/data/collect_raw_images.py`, `src/data/audit_image_bbox_pairs.py` | raw images, annotations | `data/metadata/image_manifest.csv`, `image_sources.csv` | project-3 已獨立保存正式需要影像。 |
| 2 | BBox ROI crop | `src/data/crop_bbox_rois.py` | annotation BBox + raw CXR | ROI crop outputs | ROI 階段用 BBox 製作 ROI；最終 Full-image Demo 不用 BBox/ROI。 |
| 3 | ROI 224 與稽核 | `src/data/create_roi_224_master_dataset.py`, `src/data/finalize_roi_224_dataset.py`, `src/data/prepare_model_inputs_224.py` | ROI crops/manual review | ROI 224 manifests/reports | 正式 ROI original corpus 4546 rows。 |
| 4 | Balanced ROI 4,725 | `src/audit_balanced_roi_and_build_manifest.py` | balanced ROI folder | `outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv` | 五類各 945；4256 original + 469 brightness augmented。 |
| 5 | Source-level grouped split | `src/create_phase2_grouped_split.py` | balanced/original ROI manifests | train/val/test ROI manifests | train 3770, val 454, test 454；val/test augmented count 0。 |
| 6 | RAD-DINO CLS cache | `src/cache_raddino_teacher_features.py` | balanced ROI manifest | `teacher_features.pt` | frozen `microsoft/rad-dino`, CLS feature shape `[4725, 768]`。 |
| 7 | RAD-DINO patch cache | `src/cache_raddino_teacher_patch_features.py` | balanced ROI manifest | patch feature cache | frozen patch features shape `[4725, 768, 7, 7]`。 |
| 8 | CLS distillation | `src/train_convnext_tiny_phase1_distillation.py` | ROI + CLS cache | CLS distilled backbone | MSE/L2 feature loss；class labels not used；30 epochs。 |
| 9 | Patch distillation | `src/train_convnext_tiny_phase1_patch_distillation.py` | ROI + patch cache | patch-distilled backbone | best monitor MSE `0.0002644237`, cosine `0.89846133`。 |
| 10 | ROI Phase 2 classification | `src/train_phase2_convnext_tiny_finetune.py` | grouped ROI split | three checkpoints/metrics | single-label Softmax/CrossEntropyLoss。 |
| 11 | 3-model comparison | `src/compare_baseline_cls_patch.py` | paired predictions | `overall_metrics_comparison.csv`, bootstrap CSV | principal bootstrap CIs include zero；不可宣稱全面顯著勝出。 |
| 12 | Full-image dataset | `src/prepare_full_image_224_multilabel_dataset.py` | full raw images + annotations | full-image train/val/test manifests | 590 images, split 472/59/59, five-class multilabel。 |
| 13 | ROI Patch transfer | `src/train_full_image_224_multilabel_patch_transfer.py` | ROI Patch checkpoint + full-image manifests | full-image checkpoint/threshold/metrics | old ROI head discarded，新 5-label head；test macro-F1 `0.786509`。 |
| 14 | Ground Truth catalog | `src/build_full_image_ground_truth_catalog.py` | full-image manifest | `full_image_ground_truth_manifest.csv` | 590 catalog rows；Demo lookup only。 |
| 15 | Gradio + Ollama + PDF | `app_full_image_multilabel_ollama_gradio.py` | single uploaded image | session JSON/CSV/MD/PDF | final Demo uses Full-image ConvNeXt-Tiny only；no YOLO/ROI/Softmax。 |

## 公平性與限制

ROI 與 Full-image 任務不同：ROI 是單標籤，Full-image 是多標籤；不可直接把 ROI Accuracy 和 Full-image exact subset accuracy 解讀成同一指標。`research_conclusion.md` 指出三模型整體差異小且 bootstrap CI 跨 0，Patch Proposed 只有疾病特異性 point estimate 改善，未形成全面顯著勝出結論。
