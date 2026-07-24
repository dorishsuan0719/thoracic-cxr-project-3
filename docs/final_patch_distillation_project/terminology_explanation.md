# 術語解釋

| 術語 | 中文名稱 | 本專案中的簡單解釋與用途 | 常見誤解 |
|---|---|---|---|
| ROI | 感興趣區域 | 由 Ground Truth BBox 從胸腔 X 光裁出的病灶區域，是分類模型的輸入單位。 | ROI 不是完整 X 光，也不代表模型完成了病灶偵測。 |
| BBox | 邊界框 | 以 `x_min, y_min, x_max, y_max` 表示病灶位置；本專案 `margin_ratio = 0`。 | BBox 不是 segmentation mask，也不是模型自己預測的框。 |
| Teacher | 教師模型 | `microsoft/rad-dino`，提供 CLS 或 patch 特徵作為學生學習目標。 | Teacher 不參與 Phase 2 分類更新，也未用本專案標籤 fine-tune。 |
| Student | 學生模型 | ConvNeXt-Tiny，Phase 1 學教師表徵，Phase 2 再做五類分類。 | Student 較小不代表一定比 Teacher 差，也不保證蒸餾後一定勝過 baseline。 |
| Knowledge distillation | 知識蒸餾 | 讓學生模仿教師特徵，而非只用 hard labels 訓練。 | 蒸餾成功不等於下游 accuracy 必然提高。 |
| Backbone | 特徵抽取骨幹 | ConvNeXt-Tiny 的卷積網路主體，輸出全域或 7 x 7 特徵。 | Backbone 不包含本研究最後的五類 classification head。 |
| Classification head | 分類頭 | Global average pooling 後接 dropout 0.2 與 `Linear(768,5)`。 | 分類頭不是教師投影器，也不是 BBox detector。 |
| CLS token | 全域分類 token | RAD-DINO 將整張 ROI 摘要成一個 768 維向量，供 CLS 蒸餾。 | CLS token 不是某個影像位置，也不是 class label。 |
| Patch embedding | 區塊嵌入 | 每個 14 x 14 pixel patch 被表示成 768 維特徵。 | 14 x 14 是每個 patch 的像素尺寸，不是整張圖只有 14 x 14 個 patches。 |
| Feature map | 特徵圖 | 帶有 channel 與空間網格的 tensor；Patch teacher/student 對齊為 `[B,768,7,7]`。 | Feature map 不是可直接當原始灰階影像解讀的像素圖。 |
| L2 normalization | L2 正規化 | 將特徵向量除以其 L2 norm，使比較聚焦方向而非尺度。 | 這不是對輸入圖片做 mean/std normalization。 |
| MSE | 均方誤差 | Phase 1 真正被最佳化的 loss，衡量正規化後教師與學生特徵差距。 | MSE 很低不代表分類錯誤率一定低。 |
| Cosine similarity | 餘弦相似度 | 監測教師與學生特徵方向相似程度；Patch 最終約 0.8985。 | 本研究沒有直接把 cosine 當 optimization loss。 |
| Manifest | 資料清單 | 記錄每張 ROI、類別、來源與 feature_index，鎖定快取和樣本順序。 | Manifest 不是訓練結果，也不是可任意重新排序的附屬檔。 |
| feature_index | 特徵索引 | `0...4724`，將 manifest 每列與 teacher cache tensor 對應。 | 它不是 class_id，也不是 annotation_index。 |
| SHA256 | 雜湊指紋 | 用來確認 manifest 內容未改變；本專案固定 hash 可驗證快取對應。 | SHA256 不能證明資料科學上正確，只能確認內容一致性。 |
| grouped split | 群組切分 | 以 source image_id 為群組做 8:1:1，確保同一 full image 的 ROI 不跨 split。 | 不是把每張 ROI 各自隨機分配。 |
| data leakage | 資料洩漏 | 訓練資料資訊不當進入 validation/test；本研究跨 split leakage 為 0。 | Phase 1 使用未標籤 test 來源屬 transductive 設計，需揭露，但不同於標籤洩漏。 |
| fine-tuning | 微調 | Phase 2 使用五類標籤更新 backbone 與 classification head。 | Phase 1 是無標籤 feature distillation，不是 supervised fine-tuning。 |
| Validation | 驗證集 | 用於 early stopping 與最佳 checkpoint 選擇，共 454 ROI。 | Validation 不是最終 Test，不能拿來做最終研究結論。 |
| Test | 測試集 | 模型定版後只評估一次，共 454 ROI、59 source images。 | 不應反覆看 Test 後再調模型或 threshold。 |
| Accuracy | 正確率 | 所有 ROI 中分類正確的比例。 | 類別不平衡或錯誤成本不同時，Accuracy 不能單獨代表完整表現。 |
| Precision | 精確率 | 被模型判為某類的 ROI 中，真正屬於該類的比例。 | Precision 高不表示漏診少；漏診較直接反映在 Recall。 |
| Recall | 召回率 | 真正屬於某類的 ROI 中，被模型正確找出的比例。 | Recall 高可能伴隨較多 false positives。 |
| F1 | F1 分數 | Precision 與 Recall 的調和平均，用於平衡兩者。 | F1 不包含 true negatives，也不等同 AUROC。 |
| Macro-F1 | 宏平均 F1 | 先算每類 F1 再等權平均；每個疾病同等重要。 | 不會依各類樣本數加權。 |
| Weighted-F1 | 加權 F1 | 依各類 support 對 F1 加權，樣本較多類別影響較大。 | 高於 Macro-F1 不代表每個少數類別都改善。 |
| AUROC | ROC 曲線下面積 | 衡量不同 threshold 下正類排序能力；本研究報告 one-vs-rest macro average。 | AUROC 高不代表固定 threshold 的 F1 或 calibration 一定好。 |
| confusion matrix | 混淆矩陣 | 列出真實類別和預測類別的計數，用來找出 class 2/4 錯誤方向。 | 不能只看對角線比例而忽略每類 support。 |
| bootstrap | 自助重抽樣 | 以 59 個 source-image clusters 做 10,000 次 paired resampling，估計模型差值不確定性。 | 本研究不是把 454 個 ROI 當成完全獨立樣本重抽。 |
| confidence interval | 信賴區間 | 模型差值的 95% 區間；跨 0 表示目前資料不能排除無差異。 | 跨 0 不代表兩模型完全相同，只代表證據不足。 |
| McNemar test | McNemar 檢定 | 比較兩模型在同一 ROI 上的成敗差異，作為補充統計。 | ROI-level McNemar 未處理同 source image 內相關性，因此不是主要推論。 |
| early stopping | 提前停止 | Validation 指標長時間未改善時停止訓練，降低過擬合與浪費。 | 最早停止的模型不一定最好，也不等於收斂到全域最佳。 |
| checkpoint | 模型檢查點 | 儲存特定 epoch 權重；正式 Test 使用既定 validation criterion 選出的最佳版本。 | 不能根據 Test 表現改選 checkpoint。 |
| LayerNorm | 層正規化 | ConvNeXt 最後特徵的正規化層；三模型初始化來源不完全相同。 | 它不是輸入影像 normalization；Patch 的預設初始化是潛在混淆因子。 |
| transductive training | 傳導式訓練 | Phase 1 使用全 4,725 張未標籤 ROI，包括未來 validation/test 來源，但不使用其標籤。 | 不是標籤洩漏，但不能與完全 train-only 的 inductive 設定混為一談。 |

