# 老師會議摘要

## 題目

**以 RAD-DINO 局部特徵蒸餾提升 ConvNeXt-Tiny 胸腔 X 光病灶 ROI 分類之研究**  
RAD-DINO Local Feature Distillation for ConvNeXt-Tiny Classification of Chest X-ray Lesion ROIs

## 一句話研究問題

把 RAD-DINO 的全域 CLS 或 7 x 7 局部 patch 表徵蒸餾到 ConvNeXt-Tiny，是否能在相同資料、切分與訓練條件下，提高五類胸腔 X 光病灶 ROI 分類表現？

## 資料與實驗設計

- 正式資料：590 張 full images、4,546 筆 Ground Truth BBox。
- 五類：Aortic enlargement、Cardiomegaly、Pleural thickening、Pulmonary fibrosis、Pleural effusion。
- 平衡 ROI：五類各 945 張，共 4,725 張 224 x 224 灰階影像。
- 前處理：BBox 原始座標裁切、`margin_ratio = 0`、保持比例 resize、黑色 padding。
- Grouped 8:1:1：依 source image_id 切分；source images 472/59/59，ROI 3,770/454/454。
- Leakage：六種跨 split 檢查均為 0；validation/test 不含 augmentation。
- 比較模型：ImageNet Baseline、RAD-DINO CLS distilled、RAD-DINO Patch distilled。
- 三模型共用 split、augmentation、optimizer、LR、batch size、early stopping、checkpoint criterion 與 test-once 流程。

## 蒸餾是否成功

CLS 教師快取為 [4725, 768]；CLS Phase 1 最佳 normalized MSE 為 0.0003958、cosine similarity 為 0.8480。

Patch 教師由 RAD-DINO 37 x 37 patch grid 經 adaptive average pooling 轉成 [4725, 768, 7, 7]；Patch Phase 1 最佳 normalized MSE 為 0.0002644、cosine similarity 為 0.8985。這表示學生確實學到教師的局部表徵。

## Test 結果

| Metric | Baseline | CLS | Patch |
|---|---:|---:|---:|
| Loss | 0.6040 | 0.5315 | **0.5214** |
| Accuracy | **0.7996** | 0.7930 | **0.7996** |
| Macro F1 | **0.8080** | 0.8017 | 0.8053 |
| Weighted F1 | 0.7904 | 0.7844 | **0.7918** |
| Macro AUROC | 0.9484 | **0.9507** | 0.9506 |

整體指標非常接近，沒有單一模型全面領先。Baseline 的 macro F1 最佳；Patch 的 loss 最低、weighted F1 最高；CLS 的 macro AUROC 略高。

## 最重要的類別現象

- Pleural thickening F1：Baseline 0.5455 -> Patch 0.6019。
- Pleural effusion F1：Baseline 0.6556 -> Patch 0.5793。
- Patch confusion matrix 中，class 2 正確數由 51 增至 65，class 2 -> 4 由 40 降至 22。
- 同時 class 4 正確數由 59 降至 42，class 4 -> 2 由 20 增至 37。

因此 Patch 蒸餾主要造成 class 2/4 決策邊界重新分配，而不是一致提升五類。

## 統計結論

以 59 個 test source-image clusters 做 10,000 次 paired bootstrap，三組模型配對在 accuracy、macro F1、weighted F1、macro AUROC 的 12 個 95% CI 全部跨 0。McNemar 檢定經 Holm 校正後也全部不顯著。

**目前不能宣稱 CLS 或 Patch 蒸餾穩定優於 ImageNet baseline。** 可以支持的結論是：Patch feature alignment 成功，且產生明確但類別特異的錯誤型態改變。

## 需要主動揭露的限制

1. Test 只有 59 個 source-image clusters，統計檢定力有限。
2. Phase 1 使用全部 4,725 張未標籤 ROI，屬 transductive setting。
3. Patch checkpoint 未包含 final LayerNorm；Patch Phase 2 的該層為 torchvision 預設初始化，是 CLS/Patch 比較的潛在混淆因子。
4. 目前只有 seed 42，尚未做多 seed 與外部資料驗證。
5. ROI 分類結果不能直接等同 full-image detection 或臨床診斷效益。

## 建議下一步

優先順序建議為：

1. 修正 Patch checkpoint，包含 final LayerNorm，完成公平消融。
2. 以 train-only unlabeled data 重做 inductive Phase 1。
3. 執行 3 至 5 個 seeds，建立穩定性區間。
4. 對 class 2/4 錯誤案例做 BBox 尺度與影像特徵分析。
5. 測試多層 feature matching 或 class-aware contrastive loss。

## 老師可能追問

- 為何蒸餾 alignment 很好，分類卻沒有全面變好？  
  表徵相似不保證教師特徵中的所有方向都對目前五類決策有利；MSE 也不直接最佳化類別邊界。

- 為何用 clustered bootstrap？  
  同一 full image 可產生多個 ROI，ROI 並非完全獨立；以 source image 為 cluster 比 ROI-level resampling 更符合資料結構。

- Patch 模型算成功嗎？  
  Representation transfer 成功；整體分類優勢尚未被證明。兩者需分開回答。

