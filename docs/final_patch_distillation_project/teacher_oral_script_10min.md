# 10 分鐘口頭報告稿

老師好，我的研究題目是「以 RAD-DINO 局部特徵蒸餾提升 ConvNeXt-Tiny 胸腔 X 光病灶 ROI 分類之研究」。英文題目是 RAD-DINO Local Feature Distillation for ConvNeXt-Tiny Classification of Chest X-ray Lesion ROIs。

## 研究動機

胸腔 X 光雖然取得容易，但病灶可能很小、對比低，而且不同疾病在局部區域的表現相似。RAD-DINO 是針對放射影像預訓練的基礎模型，能提供醫學領域表徵，但模型較大。我的目標是把 RAD-DINO 的知識轉移到較輕量的 ConvNeXt-Tiny。

一般蒸餾常用 logits 或單一 CLS embedding。但我的資料是 Ground Truth BBox ROI，因此我提出一個進一步的問題：如果直接對齊局部 patch feature map，是否比只對齊全域向量更能保留病灶紋理與空間訊息？

## 資料流程

正式資料來自 VinDr-CXR，包含 590 張 full images 和 4,546 筆有效 Ground Truth BBox。五類是 Aortic enlargement、Cardiomegaly、Pleural thickening、Pulmonary fibrosis 和 Pleural effusion。五類各有 350 個 class-image pairs；因為允許 multi-label full image，所以 1,750 個 class-image pairs 可以由 590 張 unique images 組成。

每個 BBox 先在原始尺寸影像上依像素座標裁切，margin ratio 是 0。接著保持長寬比縮放，再以像素值 0 的黑色 padding 補成 224 x 224，不直接拉伸。原始 ROI 是 4,546 張；經固定 seed 的下採樣與 brightness augmentation 後，五類各 945 張，共 4,725 張。469 張 augmentation 只放在 train。

我特別避免 source leakage。切分不是以 ROI 為單位，而是以 source image_id 做 grouped 8:1:1。590 張 source images 分成 472、59、59；ROI 是 3,770、454、454。六種 leakage 檢查全部為 0，validation 和 test 沒有 augmentation。

## 三個比較模型

三個模型的 Phase 2 架構完全一樣：ConvNeXt-Tiny backbone、global average pooling、768 維特徵、dropout 0.2，再接五類 linear classifier。

第一個是 ImageNet pretrained baseline。第二個是 RAD-DINO CLS distilled model。第三個是 RAD-DINO Patch distilled model。除了 backbone initialization，三者使用完全相同的 split、augmentation、optimizer、learning rate、batch size、early stopping、checkpoint 選擇和 test-once 程序。公平性稽核 57 項都通過。

## CLS 蒸餾

CLS teacher cache 形狀是 [4725, 768]。RAD-DINO 使用官方 AutoImageProcessor，教師設成 frozen、eval 和 inference mode。學生與教師特徵先做 L2 normalization，再最小化 normalized MSE；cosine similarity 只用來監測。30 個 epochs 後，最佳 MSE 是 0.0003958，cosine 是 0.8480。

## Patch 蒸餾

Patch 路線保留空間資訊。RAD-DINO 的輸入是 518 x 518，patch size 是 14 x 14，所以得到 37 x 37、共 1,369 個 patch tokens。移除 special token 後重排成 [B, 768, 37, 37]，再用 adaptive average pooling 轉成 [B, 768, 7, 7]，和 ConvNeXt-Tiny 的最後 feature map 對齊。

完整 patch cache 是 [4725, 768, 7, 7]，float32 約 711 MB，沒有 NaN、Inf 或 zero norm。Patch Phase 1 最佳 epoch 是 84，normalized MSE 是 0.0002644，cosine similarity 是 0.8985。這證明局部 feature alignment 確實成功。

## 整體分類結果

Test 共 454 個 ROI。Baseline 的 accuracy 是 0.7996、macro F1 是 0.8080。CLS 的 accuracy 是 0.7930、macro F1 是 0.8017。Patch 的 accuracy 是 0.7996、macro F1 是 0.8053。

不同指標有不同領先者。Patch 的 test loss 最低，0.5214，weighted F1 最高，0.7918；CLS 的 macro AUROC 略高，0.9507；Baseline 則有最高 macro precision、macro recall 和 macro F1。差異都很小，因此不能只挑一個指標宣稱某模型全面較好。

## 類別分析

Aortic enlargement 三模型 F1 都是 0.9872。Cardiomegaly 的 CLS 和 Patch 是 0.9935。Pulmonary fibrosis 也幾乎相同。

真正明顯的差異在 Pleural thickening 和 Pleural effusion。Pleural thickening 的 F1，Baseline 是 0.5455，Patch 提升到 0.6019；Pleural effusion 則從 0.6556 降到 0.5793。

看 confusion matrix，Patch 把 Pleural thickening 的正確數從 51 提高到 65，class 2 被錯判成 class 4 從 40 降到 22。但 Pleural effusion 正確數從 59 降到 42，class 4 被錯判成 class 2 從 20 增到 37。這是一個清楚的 trade-off：局部特徵讓模型更偏向辨識 class 2，但犧牲部分 class 4。

## 統計分析

同一張 full image 可能產生多個 ROI，所以不能假設 454 個 ROI 完全獨立。我以 59 張 test source images 為 clusters，做固定 seed 42、10,000 次 paired bootstrap，比較 accuracy、macro F1、weighted F1 和 macro AUROC。

三組模型配對共 12 個 confidence intervals 全部跨 0。ROI-level exact McNemar test 經 Holm 校正後也全部不顯著。因此，在目前 test size 下，沒有統計證據支持任何蒸餾模型對 baseline 有穩定整體優勢。

## 效率與限制

Phase 2 的 Baseline、CLS、Patch 分別訓練 25、19、14 epochs，wall time 約 817、620、458 秒。Patch fine-tuning 較快，但前面還有 teacher cache 和 84 epochs Phase 1；累積 epoch time 約 64.35 分鐘，所以完整成本必須一起計算。

研究有四個主要限制。第一，test 只有 59 個 source clusters。第二，Phase 1 使用全部未標籤 ROI，屬 transductive setting。第三，Patch checkpoint 未包含 final LayerNorm，Patch Phase 2 的該層是預設初始化，可能形成混淆。第四，目前只有 seed 42，尚未做外部驗證。

## 結論

我的最終結論分兩層。第一，representation learning 層面，RAD-DINO Patch 特徵已成功蒸餾到 ConvNeXt-Tiny。第二，下游 classification 層面，三模型整體相近，Patch 的優勢主要集中在 Pleural thickening，並伴隨 Pleural effusion 下降；統計上無法證明全面優於 baseline。

下一步會先補做包含 final LayerNorm 的公平 Patch 消融，再做 train-only inductive distillation、多 seed、class 2/4 錯誤分析與外部資料驗證。這樣才能判斷局部蒸餾的效果是否穩定，以及應如何改善它的類別平衡。謝謝老師。
