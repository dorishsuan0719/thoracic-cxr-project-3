# 5 分鐘口頭報告稿

老師好，我的專題題目是「以 RAD-DINO 局部特徵蒸餾提升 ConvNeXt-Tiny 胸腔 X 光病灶 ROI 分類之研究」。

這個研究想回答一個問題：大型胸腔 X 光基礎模型 RAD-DINO 已經學到醫學影像表徵，我能不能把這些知識轉移到較小的 ConvNeXt-Tiny，而且局部 patch 特徵是否比單一全域 CLS 特徵更適合病灶 ROI？

資料方面，我從 VinDr-CXR 建立正式五類資料集，包含 590 張 full images 和 4,546 個 Ground Truth BBox。五類是 Aortic enlargement、Cardiomegaly、Pleural thickening、Pulmonary fibrosis 和 Pleural effusion。每個 BBox 先依原始像素座標裁切，再保持比例縮放並用黑色 padding 補成 224 x 224。平衡後每類 945 張，共 4,725 張 ROI。

切分時不是隨機切 ROI，而是依 source image_id 做 grouped 8:1:1。這很重要，因為同一張 X 光可能產生多個 ROI。如果分到不同 split，模型可能在 test 看過同一個人的影像背景。最後 train、validation、test 分別是 3,770、454、454 張 ROI，六種 leakage 檢查都是 0。

我比較三個 ConvNeXt-Tiny。第一個是 ImageNet pretrained baseline。第二個先學 RAD-DINO 的 768 維 CLS 全域特徵。第三個學 RAD-DINO 的局部 patch 特徵。RAD-DINO 對 518 x 518 輸入會形成 37 x 37 patch grid，我把它池化成 7 x 7，對齊 ConvNeXt-Tiny 最後一層的 feature map。

Patch 蒸餾本身是成功的。最佳 epoch 的 normalized MSE 是 0.000264，cosine similarity 是 0.8985，代表學生的局部表徵確實接近教師。

但是，下游分類結果沒有出現全面勝利。Baseline 的 macro F1 最高，是 0.8080；Patch 是 0.8053。Patch 的 test loss 最低，是 0.5214，而且 weighted F1 最高，是 0.7918；CLS 的 macro AUROC 略高，是 0.9507。三者整體非常接近。

最值得討論的是 Pleural thickening 和 Pleural effusion。Patch 讓 Pleural thickening F1 從 0.5455 提升到 0.6019，但 Pleural effusion 從 0.6556 降到 0.5793。Confusion matrix 顯示，Patch 減少 class 2 被判成 class 4，卻增加 class 4 被判成 class 2。也就是局部蒸餾改變了這兩類的決策平衡，而不是同步改善。

統計上，我以 59 個 test source images 為 cluster 做 10,000 次 paired bootstrap。三組模型、四項主要指標的所有 95% confidence intervals 都跨過 0；McNemar 檢定經 Holm 校正後也不顯著。所以我不能宣稱 Patch 或 CLS 穩定優於 baseline。

我的結論是：RAD-DINO patch feature 可以成功蒸餾到 ConvNeXt-Tiny，並且對特定類別產生明確影響，但目前沒有證據支持整體五類分類具有穩定、全面的優勢。

下一步我會優先修正 Patch checkpoint 未包含 final LayerNorm 的問題，改用 train-only 資料做完全 inductive 蒸餾，並執行多 seed 和 class 2/4 錯誤案例分析。謝謝老師。

