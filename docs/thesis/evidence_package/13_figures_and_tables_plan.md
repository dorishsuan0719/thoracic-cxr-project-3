# 13 Figures And Tables Plan

| 編號 | 圖/表名稱 | 用途 | 現有來源 | 備註 |
|---|---|---|---|---|
| Table 1 | 正式五類與任務定義 | 避免 No Finding/Softmax/Sigmoid 混淆 | `dataset_metadata.json`, `report_prompt.py` | 論文第二/三章。 |
| Table 2 | ROI original 與 balanced class counts | 資料平衡證據 | `05_verified_dataset_counts.csv` | 含 4,546 original ROI、4,725 balanced ROI。 |
| Table 3 | ROI phase2 split counts | Split 與 leakage 控制 | `phase2_split/*.csv` | val/test 無 brightness augmented。 |
| Table 4 | RAD-DINO teacher cache shapes | 蒸餾輸入證據 | `teacher_feature_metadata.json`, `teacher_patch_feature_metadata.json` | CLS `[4725,768]`, patch `[4725,768,7,7]`。 |
| Table 5 | ROI three-model metrics | 主要 ROI 結果 | `overall_metrics_comparison.csv` | Baseline/CLS/Patch。 |
| Table 6 | Bootstrap CI | 統計限制 | `cluster_bootstrap_results.csv` | CI includes zero。 |
| Table 7 | Full-image dataset counts | Full-image 實驗資料 | `split_leakage_audit.json` | train/val/test 472/59/59。 |
| Table 8 | Full-image test metrics | final Demo 模型性能 | `test_metrics.json`, `per_class_test_metrics.csv` | macro/micro F1、per-label 指標。 |
| Table 9 | Validation thresholds | 多標籤決策規則 | `validation_selected_thresholds.json` | validation only。 |
| Figure 1 | Overall pipeline diagram | 方法總覽 | 依 `04_verified_research_pipeline.md` 重製 | 標示 ROI vs full-image。 |
| Figure 2 | ROI confusion matrices | ROI 錯誤分析 | `outputs/raddino_convnext_tiny_three_model_comparison_seed42/confusion_matrices/` | 使用既有 CSV/PNG。 |
| Figure 3 | ROI per-class F1 bar chart | 疾病別比較 | `per_class_f1_comparison.csv` | Pleural thickening/effusion 重點。 |
| Figure 4 | Full-image per-label metrics | 多標籤錯誤分析 | `per_class_test_metrics.csv` | 可重畫為 bar chart。 |
| Figure 5 | Gradio Demo screenshot | 系統展示 | 使用者截圖或既有 demo assets | 不偽裝醫療診斷。 |
| Figure 6 | Session output example | PDF/MD 報告流程 | `outputs/full_image_multilabel_gradio_demo/sessions/` | 注意匿名化。 |

## 撰寫提醒
- 若重畫圖表，需從正式 CSV/JSON 讀值。
- Bootstrap CI 跨 0 時，只能寫「未支持統計顯著差異」。
- Full-image exact subset accuracy 不是單標籤 Accuracy。
