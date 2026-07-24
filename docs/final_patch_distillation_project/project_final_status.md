# 專案最終狀態

## 總覽

- 專案主線：VinDr-CXR Ground Truth BBox ROI -> 224 x 224 平衡資料 -> RAD-DINO teacher cache -> ConvNeXt-Tiny CLS/Patch 蒸餾 -> 共用 grouped Phase 2 -> 三模型公平比較。
- 最終資料：590 張 full images、4,546 筆 BBox、4,725 張平衡 ROI。
- 最終比較：ImageNet Baseline、RAD-DINO CLS distilled、RAD-DINO Patch distilled。
- 總稽核狀態：**PASS**。
- 舊資料與既有實驗 artifacts：本報告階段均為唯讀，未修改。

| 階段 | 狀態 | 主要輸入 | 主要輸出 | 關鍵稽核或指標 | 修改既有資料 |
|---|---|---|---|---|---|
| 1. 原始 ROI 建立 | PASS | `data/raw/images`、`data/raw/annotations/annotations.csv` | `data/processed/bbox_crops`、`bbox_crops_224` | 590 full images、4,546 BBox/crops；五類 772/783/1118/1062/811 | 否；輸出為另建 processed artifacts |
| 2. 平衡資料建立 | PASS | 正式 224 x 224 ROI | `outputs/roi_balanced_224/balanced_945_seed42` | 五類各 945，共 4,725；原始 4,256、brightness augmented 469 | 否 |
| 3. Phase 0-A Manifest audit | PASS | 4,725 張平衡 ROI | `outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv` 與 audit files | rows 4,725；index 0-4724；missing/unreadable/wrong size/wrong mode 皆 0；SHA256 `796f...5233` | 否 |
| 4. CLS Teacher cache | PASS | roi_manifest、`microsoft/rad-dino` | `teacher_features.pt` 與 metadata/audit | `[4725,768]`、float32 CPU；NaN/Inf/zero norm 0 | 否 |
| 5. CLS Phase 1 | PASS | CLS teacher cache、ConvNeXt-Tiny | `phase1_distillation` distilled backbone | 30 epochs；best epoch 30；MSE 0.0003958；cosine 0.8480；未使用 labels | 否 |
| 6. Grouped Phase 2 split | PASS | roi_manifest、source_image_id | `phase2_split` | source 472/59/59；ROI 3770/454/454；六種 leakage 0；val/test augmentation 0 | 否 |
| 7. CLS Phase 2 | PASS | CLS distilled backbone、共用 split/config | `phase2_proposed_distilled` | 19 epochs；best epoch 9；Test macro F1 0.8017；Test evaluation count 1 | 否 |
| 8. Baseline Phase 2 | PASS | ImageNet ConvNeXt-Tiny、共用 split/config | `phase2_baseline_imagenet` | 25 epochs；best epoch 19；Test macro F1 0.8080；Test evaluation count 1 | 否 |
| 9. Baseline vs CLS 比較 | PASS | 兩模型 predictions/metrics | 既有 comparison artifacts | paired rows 454；公平性設定一致；整體差異未顯著 | 否 |
| 10. Patch shape smoke test | PASS | RAD-DINO、ConvNeXt-Tiny、sample ROI | shape validation artifacts | RAD-DINO 37 x 37 -> 7 x 7；Student `[B,768,7,7]`；無 NaN/Inf/OOM | 否 |
| 11. Patch Teacher cache | PASS | 4,725 ROI、RAD-DINO patch tokens | `phase0_patch_teacher_cache` | `[4725,768,7,7]`、約 711 MB；NaN/Inf/zero spatial norm 0 | 否 |
| 12. Patch Phase 1 | PASS | Patch teacher cache、ConvNeXt-Tiny | `phase1_patch_distillation` | 84 epochs；best 84；MSE 0.0002644；cosine 0.8985；未使用 labels | 否 |
| 13. Patch Phase 2 | PASS | Patch distilled backbone、共用 split/config | `phase2_proposed_patch_distilled` | 14 epochs；best epoch 4；Test macro F1 0.8053；Test evaluation count 1 | 否 |
| 14. 三模型公平比較 | PASS | 三模型 Test predictions/metrics | `raddino_convnext_tiny_three_model_comparison_seed42` | 57 fairness checks；非允許差異 0；各 454 predictions；paired key error 0 | 否 |
| 15. 統計分析與最終結論 | PASS | paired predictions、source clusters、LayerNorm audit | comparison report、research conclusion、本文件集 | 59 clusters；10,000 bootstrap；12 個 CI 全跨 0；Holm p 全 1.0；LayerNorm/transductive 限制已揭露 | 否 |

## 最終 Test 指標

| Metric | Baseline | CLS | Patch |
|---|---:|---:|---:|
| Loss | 0.60400416 | 0.53153514 | 0.52136961 |
| Accuracy | 0.79955947 | 0.79295154 | 0.79955947 |
| Macro-F1 | 0.80799958 | 0.80174819 | 0.80525937 |
| Weighted-F1 | 0.79042816 | 0.78440979 | 0.79178031 |
| Macro-AUROC | 0.94842022 | 0.95066454 | 0.95058078 |

## 仍需補強的研究工作

- Patch checkpoint 應包含 final LayerNorm，再做控制變項消融。
- Phase 1 應增加 train-only inductive 版本。
- 應執行多 seeds 與外部資料驗證。
- Class 2/4 應做病例與 BBox 尺度分層分析。

## 本次文件整理邊界

本次只新增 `docs/final_patch_distillation_project` 內的研究文件與稽核檔。未重新訓練、未重新評估 Validation/Test、未執行圖片 inference、未修改 threshold、split、checkpoint、metrics、來源 ROI、teacher cache、training scripts、comparison scripts 或既有 outputs。

