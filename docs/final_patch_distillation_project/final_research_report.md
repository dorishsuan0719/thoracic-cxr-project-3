# 以 RAD-DINO 局部特徵蒸餾提升 ConvNeXt-Tiny 胸腔 X 光病灶 ROI 分類之研究

**RAD-DINO Local Feature Distillation for ConvNeXt-Tiny Classification of Chest X-ray Lesion ROIs**

## 摘要

本研究探討是否能以胸腔 X 光專用基礎模型 RAD-DINO 作為教師，透過無標籤知識蒸餾改善輕量化 ConvNeXt-Tiny 對五類胸腔病灶 ROI 的分類能力。資料由 VinDr-CXR Ground Truth BBox 建立，正式 full-image 子集合包含 590 張影像與 4,546 筆 BBox；經原始 ROI 裁切、等比例縮放、黑色 padding 與類別平衡後，共得到 4,725 張 224 x 224 灰階 ROI。為避免同一來源影像洩漏，分類資料以 source image_id 進行 grouped 8:1:1 切分，得到 3,770/454/454 張 train/validation/test ROI，且六種跨 split leakage 檢查皆為 0。

研究比較三種 ConvNeXt-Tiny：ImageNet baseline、RAD-DINO CLS 全域特徵蒸餾、RAD-DINO 7 x 7 patch 局部特徵蒸餾。Patch 教師快取形狀為 [4725, 768, 7, 7]；學生以 normalized patch MSE 最佳化，Phase 1 最佳 epoch 為 84，MSE 0.0002644、cosine similarity 0.8985。Phase 2 使用相同 split、資料增強、optimizer、learning rate、batch size、early stopping 與測試程序，只允許初始化來源不同。

測試結果顯示三模型整體表現接近。Baseline 的 macro F1 最高（0.8080），Patch 的 test loss 最低（0.5214）且 weighted F1 最高（0.7918），CLS 的 macro AUROC 略高（0.9507）。Patch 對 Pleural thickening 的 F1 由 baseline 的 0.5455 提升至 0.6019，但 Pleural effusion 由 0.6556 降至 0.5793，呈現 class 2/4 錯誤重新分配。以 59 個 source image cluster 進行 10,000 次 paired bootstrap，四項主要指標的所有模型差值 95% CI 均跨越 0；McNemar 檢定經 Holm 校正後亦均不顯著。因此，本研究可確認 patch-level feature alignment 成功，但不能宣稱其帶來穩定且全面的分類優勢。

## 1. 研究背景與動機

胸腔 X 光具備成本低、取得快速與臨床使用廣泛等優點，但病灶外觀常有低對比、尺度差異與類別間高度相似等特性。大型醫學影像基礎模型可提供豐富的領域知識，卻通常具有較高運算成本；若能將其表徵轉移到較小模型，便可能兼顧醫學領域特徵與部署效率。

傳統知識蒸餾常以分類 logits 或全域 CLS embedding 為目標。對 BBox ROI 而言，病灶的局部紋理、邊界和空間位置可能比單一全域向量更重要。本研究因此進一步將 RAD-DINO 的 patch tokens 重排為空間特徵圖，再池化至與 ConvNeXt-Tiny 相同的 7 x 7 網格，直接進行局部表徵對齊。

## 2. 研究問題

本研究回答四個問題：

1. RAD-DINO 的全域 CLS 特徵能否有效蒸餾至 ConvNeXt-Tiny？
2. 7 x 7 patch-level 蒸餾能否比 ImageNet baseline 與 CLS 蒸餾提供更好的五類 ROI 分類能力？
3. 若整體指標差異有限，局部蒸餾是否仍會改變特定類別的辨識與錯誤型態？
4. 三模型的點估計差異在 source-image clustered bootstrap 後是否仍獲統計支持？

## 3. 研究貢獻

1. 建立具完整追溯性的五類胸腔 X 光 ROI 資料流程，保留 full image、Ground Truth BBox、原始 ROI 與 224 x 224 模型輸入之對應。
2. 建立 RAD-DINO CLS 與 7 x 7 patch teacher feature cache，使 Phase 1 蒸餾不需在每個 epoch 重複執行教師模型。
3. 在相同 grouped split 與 Phase 2 設定下，完成 ImageNet、CLS 蒸餾與 Patch 蒸餾三模型的公平比較。
4. 使用 source-image clustered paired bootstrap 與 McNemar 檢定，避免把同一張 full image 的多個 ROI 誤當成完全獨立樣本。
5. 揭示 Patch 蒸餾的類別特異性效果：改善 Pleural thickening，但加劇其與 Pleural effusion 的部分混淆。

## 4. 資料來源與正式資料集

資料來源為 VinDr-CXR。正式 full-image 子集合包含 590 張影像，每個目標類別均有 350 個 class-image pairs；因允許 multi-label full image，五類合計 1,750 個 class-image pairs 可由 590 張 unique full images 組成。所有入選圖片均保留其五個目標類別中的全部有效 Ground Truth BBox。

五類定義如下：

| class_id | class_name |
|---:|---|
| 0 | Aortic enlargement |
| 1 | Cardiomegaly |
| 2 | Pleural thickening |
| 3 | Pulmonary fibrosis |
| 4 | Pleural effusion |

正式 annotations.csv 共 4,546 筆 BBox，五類 BBox 數依序為 772、783、1,118、1,062、811。missing image、invalid BBox、BBox 越界、完全重複 annotation 與五類外 annotation 均為 0。

## 5. ROI 前處理與平衡資料

每一筆有效 BBox 先從原始 full image 依像素座標直接裁切，`margin_ratio = 0`，不先 resize 或 letterbox。接著保持長寬比例縮放至可完整放入 224 x 224，置中後以像素值 0 的黑色 padding 補足，不裁切 ROI 內容、不做 mean/std normalization。

原始 4,546 張 ROI 經固定 seed 42 的下採樣與 brightness augmentation 建立五類各 945 張的平衡資料，共 4,725 張。其中 4,256 張來自正式原始 ROI，469 張為 brightness augmentation；augmentation 只進入訓練 split。Phase 0-A manifest 的 4,725 列與 `feature_index = 0...4724` 均連續且唯一，SHA256 為 `796f067d00bb5740a51b51292eed4acfefe9b2e84fd2eeb9b5dfd2df926d5233`。

## 6. Grouped 8:1:1 切分

切分單位為 source image_id，而非 ROI。590 張來源影像分為 train/validation/test = 472/59/59；對應 ROI 為 3,770/454/454。validation 與 test 均不含 augmented ROI。source image、ROI、原始與增強樣本相關的六種 leakage 檢查全部為 0。

Test set 各類 ROI 數為 77、78、112、106、81。因一張來源影像可能貢獻多個 ROI，統計推論以 59 個 source image clusters 為主要抽樣單位。

## 7. 模型架構

三個 Phase 2 模型均採用 ConvNeXt-Tiny backbone，接續 global average pooling、768 維特徵、dropout 0.2 與 `Linear(768, 5)` 分類頭。輸入為 224 x 224 ROI，灰階影像由既定影像處理流程轉為模型所需通道格式。

三模型唯一允許的核心差異是 backbone 初始化：

| Model | Backbone initialization |
|---|---|
| ImageNet Baseline | torchvision ImageNet pretrained ConvNeXt-Tiny |
| RAD-DINO CLS | Phase 1 CLS feature-distilled checkpoint |
| RAD-DINO Patch | Phase 1 7 x 7 patch feature-distilled checkpoint |

## 8. RAD-DINO CLS 蒸餾

CLS 教師使用官方 `microsoft/rad-dino` 與 `AutoImageProcessor`，輸入 518 x 518，教師設定為 frozen、eval 與 inference mode。快取為 CPU float32 tensor，形狀 [4725, 768]，NaN、Inf 與 zero norm 皆為 0。

學生將 ConvNeXt-Tiny 的全域 768 維特徵投影至相同維度，對教師與學生特徵做 L2 normalization，再最小化 normalized MSE；cosine similarity 僅作監測。Phase 1 使用全部 4,725 張 ROI 且不讀取類別標籤，屬於 transductive unlabeled representation distillation。最佳 epoch 30 的 MSE 為 0.0003958、cosine similarity 為 0.8480。

## 9. RAD-DINO Patch 蒸餾

RAD-DINO 對 518 x 518 輸入使用 14 x 14 patch，因此產生 37 x 37、共 1,369 個 patch tokens，另含 1 個 special token。移除 special token 後，patch embedding 重排為 [B, 768, 37, 37]，再以 `adaptive_avg_pool2d` 池化為 [B, 768, 7, 7]，與 ConvNeXt-Tiny 最終 feature map 對齊。

完整教師快取形狀為 [4725, 768, 7, 7]，float32 原始 tensor 約 711.2 MB；NaN、Inf 與 zero spatial norm 均為 0。學生僅最佳化 L2-normalized patch MSE，cosine similarity 為監測指標。最佳 epoch 84 的 MSE 為 0.0002644、cosine similarity 為 0.8985，證明局部表徵確實可被學生近似。

![Patch MSE learning curve](../../outputs/raddino_convnext_tiny_patch_experiment_seed42/phase1_patch_distillation/figures/patch_monitor_mse_curve.png)

![Patch cosine similarity curve](../../outputs/raddino_convnext_tiny_patch_experiment_seed42/phase1_patch_distillation/figures/patch_cosine_curve.png)

## 10. Phase 2 公平性設計

三模型共用完全相同的 grouped split、ROI 清單、augmentation 規則、optimizer、learning rates、batch size、early stopping、checkpoint 選擇與 test-once 程序。公平性稽核共 57 項，非允許差異為 0。每個模型只執行一次正式 test evaluation，各產生 454 筆 prediction，三模型配對鍵完整且唯一。

需要特別揭露一項限制：Baseline final LayerNorm 來自 ImageNet；CLS 模型 final LayerNorm 來自 CLS 蒸餾 checkpoint；Patch Phase 1 匯出未包含 final LayerNorm，因此 Patch Phase 2 的 final LayerNorm 為 torchvision 預設初始化。這是 CLS 與 Patch 比較中的潛在混淆因子，未來應以包含 final LayerNorm 的 Patch checkpoint 重做消融。

## 11. 整體測試結果

| Metric | Baseline | CLS distilled | Patch distilled | Point-estimate leader |
|---|---:|---:|---:|---|
| Test loss | 0.6040 | 0.5315 | **0.5214** | Patch |
| Accuracy | **0.7996** | 0.7930 | **0.7996** | Baseline / Patch |
| Macro precision | **0.8085** | 0.8011 | 0.8081 | Baseline |
| Macro recall | **0.8210** | 0.8123 | 0.8097 | Baseline |
| Macro F1 | **0.8080** | 0.8017 | 0.8053 | Baseline |
| Weighted F1 | 0.7904 | 0.7844 | **0.7918** | Patch |
| Macro AUROC | 0.9484 | **0.9507** | 0.9506 | CLS |

![Overall metrics across three models](../../outputs/raddino_convnext_tiny_three_model_comparison_seed42/figures/overall_metrics_three_models.png)

三模型沒有單一模型在所有指標上領先。Baseline 保有最高 macro F1、macro recall 與 macro precision；Patch 具有最低 loss、並列最高 accuracy 及最高 weighted F1；CLS 的 macro AUROC 僅以極小幅度領先。

## 12. Per-class 結果

| Class | Baseline F1 | CLS F1 | Patch F1 |
|---|---:|---:|---:|
| Aortic enlargement | 0.9872 | 0.9872 | 0.9872 |
| Cardiomegaly | 0.9872 | **0.9935** | **0.9935** |
| Pleural thickening | 0.5455 | 0.5417 | **0.6019** |
| Pulmonary fibrosis | **0.8646** | 0.8621 | 0.8644 |
| Pleural effusion | **0.6556** | 0.6243 | 0.5793 |

![Per-class F1 comparison](../../outputs/raddino_convnext_tiny_three_model_comparison_seed42/figures/per_class_f1_three_models.png)

Patch 蒸餾最明顯的正向變化出現在 Pleural thickening，F1 比 baseline 高約 0.0564；但 Pleural effusion F1 同時下降約 0.0762。這表示局部特徵對齊改變了模型在相鄰影像表現類別間的決策邊界，而非一致提升所有疾病。

## 13. Confusion matrix 與 class 2/4 trade-off

Patch 模型在 Pleural thickening 的正確預測由 baseline 的 51 增至 65，class 2 被誤判為 class 4 的數量由 40 降至 22；相對地，Pleural effusion 的正確預測由 59 降至 42，class 4 被誤判為 class 2 的數量由 20 增至 37。

![Patch model confusion matrix](../../outputs/raddino_convnext_tiny_three_model_comparison_seed42/figures/confusion_matrix_patch.png)

![Class 2 and class 4 confusion comparison](../../outputs/raddino_convnext_tiny_three_model_comparison_seed42/figures/class2_class4_confusion_comparison.png)

此結果不應被簡化為 Patch 模型「更好」或「更差」。較精確的描述是：Patch 蒸餾使 class 2/4 的錯誤方向重新分配，對 Pleural thickening 有利、對 Pleural effusion 不利。

## 14. 統計分析

主要推論使用 source-image clustered paired bootstrap：以 59 張 test source images 為 cluster，固定 seed 42，進行 10,000 次重抽樣，比較 accuracy、macro F1、weighted F1 與 macro AUROC。三組模型配對共 12 個差值的 95% confidence interval 全部跨越 0。

![Clustered bootstrap confidence intervals for macro F1](../../outputs/raddino_convnext_tiny_three_model_comparison_seed42/figures/cluster_bootstrap_ci_macro_f1.png)

ROI-level exact McNemar test 作為補充分析，未校正 p 值分別為 CLS vs Baseline 0.7709、Patch vs Baseline 1.0000、Patch vs CLS 0.7660；Holm 校正後皆為 1.0。由於同一 source image 可能有多個 ROI，McNemar 結果只作輔助，主要結論以 cluster bootstrap 為準。

因此，所有整體優勢都只能視為點估計差異，尚無足夠統計證據支持任一蒸餾模型穩定優於 baseline。

## 15. 訓練效率

Phase 2 的 Baseline、CLS、Patch 分別訓練 25、19、14 epochs，最佳 epoch 為 19、9、4；wall time 約為 816.9、620.0、457.8 秒。三者峰值 GPU allocated memory 都約 5.22 GiB、reserved memory 約 5.49 GiB。

![Training efficiency comparison](../../outputs/raddino_convnext_tiny_three_model_comparison_seed42/figures/training_efficiency_three_models.png)

Patch Phase 2 收斂較快，但其前置 Phase 1 成本不可忽略。84 個 epoch 記錄的累積 `epoch_total_seconds` 約 3,860.8 秒（64.35 分鐘）；final audit 中 1,774.7 秒則只代表最後一次 resumed segment，不能當作完整 Phase 1 總時間。若只看 Phase 2，Patch 最省時；若評估完整研究管線，必須把 teacher cache 與 Phase 1 蒸餾納入。

## 16. 主要發現

1. RAD-DINO 的全域與局部特徵都能被 ConvNeXt-Tiny 有效近似，Patch alignment 的最終 cosine similarity 達 0.8985。
2. 特徵對齊成功不等同於下游分類全面提升。
3. 三模型整體指標非常接近，且所有 cluster bootstrap CI 均跨 0。
4. Patch 蒸餾主要改變 Pleural thickening 與 Pleural effusion 的辨識平衡。
5. Patch Phase 2 較早達到最佳 validation checkpoint，但完整蒸餾流程有額外計算成本。

## 17. 研究限制

1. Test source images 只有 59 張，cluster-level 統計檢定力有限。
2. Phase 1 使用全部 4,725 張未標籤 ROI，包括未來 validation/test 來源，因此屬 transductive setting；雖未使用標籤，仍需與完全 inductive setting 區分。
3. Patch checkpoint 未包含 final LayerNorm，造成 CLS 與 Patch 初始化不完全對稱。
4. 只使用單一 seed 42，無法估計訓練隨機性的變異。
5. 資料來自單一資料集，尚未進行外部醫院或跨裝置驗證。
6. ROI 分類不等同於 full-image detection，也不能直接推論臨床診斷效益。
7. Class 2/4 的互換顯示單純 MSE 空間對齊可能不足以保留疾病判別方向。

## 18. 未來工作

1. 以 3 至 5 個 seeds 重複三模型訓練，回報平均值、標準差與分層統計。
2. 建立完全 inductive 的 Phase 1，只用 train source images 做無標籤蒸餾。
3. 讓 Patch checkpoint 一併匯出 final LayerNorm，消除初始化混淆。
4. 測試多層 feature matching、attention-weighted pooling、token selection 與 class-aware contrastive objectives。
5. 對 class 2/4 做錯誤案例分層、BBox 尺度分析與 radiologist review。
6. 進行外部資料集驗證、校準分析與 robustness assessment。
7. 比較 cache 成本、端到端蒸餾與實際部署 latency，以評估效益是否足以抵銷前處理成本。

## 19. 結論

本研究完成一套可追溯、無 source-image leakage 的五類胸腔 X 光 ROI 跨架構知識蒸餾實驗。RAD-DINO 7 x 7 patch 特徵能被 ConvNeXt-Tiny 成功對齊，並在 Pleural thickening 類別得到較高 F1，同時達到最低 test loss 與略高 weighted F1。然而，Baseline 仍具有最高 macro F1，Patch 對 Pleural effusion 的辨識下降，且所有 source-cluster bootstrap confidence intervals 均跨越 0。

因此，最穩健的研究結論是：**Patch feature alignment 已成功，且會產生類別特異性的決策改變；但目前證據不足以支持 RAD-DINO Patch 蒸餾對整體五類 ROI 分類具有穩定、全面且統計顯著的優勢。**

## 20. 可重現性與稽核

本報告中的資料數量、split、manifest hash、Phase 1 指標、三模型 Test metrics、per-class F1、confusion matrices、bootstrap、McNemar、訓練效率與 LayerNorm 初始化均由既有 JSON/CSV/設定檔重新讀取驗證。稽核結果記錄於 [result_verification_audit.json](result_verification_audit.json)，總狀態為 PASS。

核心圖表來源與口頭說明索引見 [core_figures_manifest.csv](core_figures_manifest.csv)。本階段未重新訓練、未重新評估 validation/test、未執行圖片 inference、未更改 split、threshold、checkpoint 或既有實驗輸出。
