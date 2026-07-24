# 老師簡報大綱

建議 15 頁，約 10 至 12 分鐘。每頁保留一個主要訊息，不在投影片塞滿實驗細節。

## Slide 1｜研究背景與題目

**建議圖表：** 胸腔 X 光 ROI、RAD-DINO、ConvNeXt-Tiny 的簡單流程示意。

**重點：**

- 胸腔 X 光病灶低對比、局部且類別間相似。
- RAD-DINO 有醫學領域表徵，但模型較大。
- 研究目標是把教師知識轉移到輕量 ConvNeXt-Tiny。

**口頭重點：** 先說明這不是單純比較兩個 classifier，而是研究全域與局部表徵蒸餾是否有效。

## Slide 2｜研究問題

**建議圖表：** 四個 RQ 的簡潔方塊圖。

**重點：**

- RQ1：CLS 蒸餾能否改善 ImageNet initialization？
- RQ2：Patch 蒸餾能否優於 CLS 蒸餾？
- RQ3：局部蒸餾是否改善特定病灶類別？
- RQ4：模型差異經 source-cluster 統計後是否仍成立？

**口頭重點：** 表徵對齊成功和分類顯著改善是兩個不同層次的問題。

## Slide 3｜資料來源與前處理

**建議圖表：** Full image -> BBox -> ROI -> 224 x 224 letterbox 流程。

**重點：**

- 590 張 full images、4,546 筆 Ground Truth BBox。
- 五類 class-image pairs 各 350。
- 原始像素 BBox 裁切，`margin_ratio = 0`。
- 保持比例 resize、黑色 padding，不直接拉伸。

**口頭重點：** 每一筆 BBox 都可追溯回 source image 與 annotation。

## Slide 4｜ROI 平衡與 Manifest

**建議圖表：** 五類各 945 張的長條圖或 manifest 流程。

**重點：**

- 4,256 張原始 ROI、469 張 brightness augmentation。
- 五類各 945 張，共 4,725 張。
- `feature_index = 0...4724` 連續且唯一。
- Manifest SHA256 鎖定資料順序與 teacher cache 對應。

**口頭重點：** Manifest 是所有後續快取、蒸餾與切分的唯一樣本索引。

## Slide 5｜研究方法總覽

**建議圖表：** Baseline、CLS、Patch 三路線並行流程圖。

**重點：**

- Baseline：ImageNet pretrained ConvNeXt-Tiny。
- CLS：對齊 RAD-DINO 768 維全域特徵。
- Patch：對齊 RAD-DINO 7 x 7 局部 feature map。
- Phase 2 使用相同分類頭與訓練設定。

**口頭重點：** 三條路線最終都回到相同 ConvNeXt-Tiny 架構，公平比較初始化效果。

## Slide 6｜CLS 特徵蒸餾

**建議圖表：** Teacher/Student `[B,768]` 對齊示意。

**重點：**

- Teacher cache `[4725,768]`，CPU float32。
- Teacher frozen、eval、inference mode。
- L2 normalization + normalized MSE。
- 最佳 MSE 0.0003958、cosine 0.8480。

**口頭重點：** 快取避免每個 epoch 重跑 RAD-DINO，也保證教師目標固定。

## Slide 7｜Patch 特徵蒸餾

**建議圖表：** 37 x 37 -> adaptive pooling -> 7 x 7 對齊圖。

**重點：**

- 14 x 14 是單一 patch 的 pixel size，不是 patch 數量。
- RAD-DINO 形成 37 x 37 = 1,369 個 patch tokens。
- Teacher `[B,768,37,37]` 池化至 `[B,768,7,7]`。
- Student 最終 feature map 也是 `[B,768,7,7]`。

**口頭重點：** Patch 路線保留空間位置，理論上更貼近局部病灶。

## Slide 8｜Patch Phase 1 收斂

**建議圖表：** `patch_monitor_mse_curve.png` 與 `patch_cosine_curve.png`。

**重點：**

- Epoch 1：MSE 0.0010347、cosine 0.6027。
- Epoch 30：MSE 0.0003487、cosine 0.8661。
- Epoch 60：MSE 0.0002798、cosine 0.8926。
- Epoch 84：MSE 0.0002644、cosine 0.8985。

**口頭重點：** Alignment 明確改善，但 alignment 不是下游分類優勢的充分條件。

## Slide 9｜Grouped Split 與公平性

**建議圖表：** source image grouped split 示意或 fairness audit 摘要。

**重點：**

- Source images 472/59/59；ROI 3,770/454/454。
- 六種 leakage 全部為 0。
- Val/Test augmented ROI 為 0。
- 57 項公平性檢查通過，三模型各 test 一次。

**口頭重點：** 同一 full image 的 ROI 不會跨 split，避免隱性資料洩漏。

## Slide 10｜整體 Test 結果

**建議圖表：** `overall_metrics_three_models.png`。

**重點：**

- Baseline macro F1 最高：0.8080。
- Patch loss 最低：0.5214；weighted F1 最高：0.7918。
- CLS macro AUROC 略高：0.9507。
- 三者 accuracy 約 0.7930 至 0.7996。

**口頭重點：** 沒有單一模型全面領先，差異都是小幅點估計。

## Slide 11｜Per-class F1

**建議圖表：** `per_class_f1_three_models.png`。

**重點：**

- Class 0 幾乎相同。
- Class 1 的 CLS/Patch 略高。
- Patch 的 class 2 F1 從 0.5455 升到 0.6019。
- Patch 的 class 4 F1 從 0.6556 降到 0.5793。

**口頭重點：** Patch 的影響是 class-specific，不是全面提升。

## Slide 12｜Confusion Matrix 與 Class 2/4 Trade-off

**建議圖表：** `confusion_matrix_patch.png`、`class2_class4_confusion_comparison.png`。

**重點：**

- Class 2 correct：51 -> 65。
- Class 2 -> 4：40 -> 22。
- Class 4 correct：59 -> 42。
- Class 4 -> 2：20 -> 37。

**口頭重點：** Patch 使決策邊界偏向 Pleural thickening，同時犧牲 Pleural effusion。

## Slide 13｜Cluster Bootstrap 統計

**建議圖表：** `cluster_bootstrap_ci_macro_f1.png`。

**重點：**

- 59 個 source-image clusters。
- 10,000 次 paired bootstrap，seed 42。
- 3 組模型配對 x 4 指標，共 12 個 CI。
- 所有 95% CI 均跨 0；Holm-adjusted McNemar p 全為 1.0。

**口頭重點：** 目前沒有統計證據支持蒸餾模型穩定優於 baseline。

## Slide 14｜限制與方法學警告

**建議圖表：** 限制矩陣或簡短圖示。

**重點：**

- 單一 seed、test source clusters 只有 59。
- Phase 1 使用全部未標籤 ROI，屬 transductive setting。
- Patch checkpoint 未包含 final LayerNorm。
- 單一資料集，未做外部驗證與 calibration。

**口頭重點：** 主動揭露 LayerNorm 與 transductive 設計，避免過度解讀。

## Slide 15｜結論與下一步

**建議圖表：** 核心結論三句與後續實驗優先順序。

**重點：**

- Patch feature alignment 成功。
- Pleural thickening 改善，但 Pleural effusion 下降。
- 整體優勢未獲統計支持。
- 下一步：LayerNorm 公平消融、inductive Phase 1、多 seed、class 2/4 分析。

**口頭重點：** 結論要保守但不否定成果，研究價值在於找出局部蒸餾的作用方式與限制。

