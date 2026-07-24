# 胸腔X光影像分類與知識蒸餾系統：論文保守初稿

本文件為依據 `docs/thesis/evidence_package/` 建立之保守初稿。所有未能由正式程式、CSV、JSON、audit 或 metrics 直接確認之內容，均以 `〔待確認〕` 或 `〔待補正式文獻〕` 標記。

# 第一章　緒論

## 1.1 研究背景與動機

胸腔 X 光影像是臨床常見且取得相對容易的影像檢查之一，常被用於觀察心胸輪廓、肺部紋理、肋膜變化與胸腔內液體等影像徵象。隨著深度學習技術發展，卷積神經網路與視覺基礎模型逐漸被應用於醫學影像分類與輔助判讀。然而，胸腔 X 光影像中的異常位置、大小與對比差異很大，若僅使用完整影像進行分類，模型可能同時受到背景、姿勢、影像品質與多重標籤共存等因素影響。

本專題以五類胸腔 X 光相關影像標籤為研究範圍，先建立 ROI 階段的五類單標籤分類流程，再將 ROI 階段的 Patch Proposed backbone 轉移至 Full-image 五類多標籤任務，最後整合 Gradio 使用者介面、本機 Ollama 說明生成、Ground Truth 比對與報告匯出。由於本研究仍屬學術專題展示，所有模型輸出均不得視為正式醫療診斷。

## 1.2 研究問題

本研究關注的核心問題包括：ROI 區域化是否有助於建立較清楚的五類影像辨識流程；RAD-DINO 的 CLS 與 patch feature 是否能作為 ConvNeXt-Tiny 的蒸餾 teacher；Patch-level 蒸餾特徵能否作為 Full-image 多標籤模型的初始化來源；以及最終 Demo 是否能在不讓大型語言模型直接判讀影像的前提下，根據本機模型輸出產生保守、可稽核的白話說明。

## 1.3 研究目的

- 建立五類 ROI 單標籤分類資料與對應之 train、validation、test 流程。
- 使用 RAD-DINO frozen teacher 建立 CLS 與 patch feature cache，並蒸餾至 ConvNeXt-Tiny student。
- 比較 ImageNet Baseline、RAD-DINO CLS Proposed 與 RAD-DINO Patch Proposed 三種 ROI 模型。
- 將 ROI Patch Proposed backbone 轉移至 Full-image 五類多標籤分類模型。
- 完成可本機展示的 Gradio Demo，呈現五類機率、Ground Truth 比對、Ollama 輔助說明與報告匯出。

## 1.4 研究範圍

本研究範圍限定於 evidence package 已驗證之資料、程式與輸出結果。ROI 階段為五類單標籤分類；Full-image 階段為五類多標籤分類。兩者任務定義、activation、loss 與評估指標不同，因此不可將 ROI Accuracy 與 Full-image Exact Subset Accuracy 直接視為同一種指標比較。

## 1.5 研究貢獻

- 整理出 ROI 到 Full-image 的可追蹤研究流程。
- 建立 RAD-DINO feature distillation 與 ConvNeXt-Tiny downstream classification 的比較框架。
- 以正式輸出結果說明 Patch Proposed 在部分指標具競爭力，但不宣稱全面或統計顯著優於 Baseline。
- 建置可重現的 Full-image 多標籤 Demo 與本機文字說明流程，並明確隔離 Ground Truth 與模型推論。

## 1.6 論文架構

本文共分六章。第一章說明研究背景、問題與目的；第二章整理相關技術背景，文獻引用於正式版本需補充來源；第三章說明資料、模型與訓練流程；第四章呈現正式實驗結果；第五章說明系統設計與 Demo 流程；第六章總結成果與限制。

# 第二章　文獻探討

## 2.1 胸腔 X 光影像分析

胸腔 X 光影像具有低成本、低輻射與高臨床可近性的特性，但其影像判讀需同時考慮解剖結構重疊、拍攝姿勢、病灶大小與影像品質。正式論文版本需補充胸腔 X 光影像分析與自動化輔助判讀之正式文獻。〔待補正式文獻〕

## 2.2 深度學習影像分類

深度學習模型可由影像資料中自動學習階層式特徵。卷積神經網路常用於局部紋理與形狀辨識，而現代架構則進一步改善表徵能力與訓練穩定性。此處需於定稿時補充深度學習影像分類與醫學影像應用之正式文獻。〔待補正式文獻〕

## 2.3 Vision Transformer 與自監督學習

自監督學習可利用大量未標註影像學習一般化特徵，Vision Transformer 類模型則常以 patch 為單位建模影像資訊。本研究使用 RAD-DINO 作為 frozen teacher，但不將其作為最終 Demo 推論模型。〔待補正式文獻〕

## 2.4 RAD-DINO 醫學影像基礎模型

RAD-DINO 在本研究中被定位為醫學影像特徵 teacher，分別提供 CLS feature 與 patch feature 作為蒸餾目標。正式版本需補充 RAD-DINO 模型來源、訓練資料與方法之文獻。〔待補正式文獻〕

## 2.5 ConvNeXt

ConvNeXt-Tiny 為本專題的 student backbone 與後續分類模型基礎。其架構可被用於接收蒸餾後的影像表徵，再於 ROI 或 Full-image 任務進行下游訓練。〔待補正式文獻〕

## 2.6 知識蒸餾

知識蒸餾旨在讓 student 模型學習 teacher 模型輸出的表徵或預測。本研究採 feature-level distillation，分為 CLS feature 與 patch feature 兩條路徑。〔待補正式文獻〕

## 2.7 CLS 與 Patch 特徵

CLS feature 代表影像層級摘要；patch feature 保留空間網格資訊。本研究比較兩者對 ROI 下游分類的影響，並將 Patch Proposed backbone 轉移至 Full-image 多標籤任務。〔待補正式文獻〕

## 2.8 多標籤分類

Full-image 階段一張完整胸腔 X 光影像可能同時包含多個目標標籤，因此使用五類多標籤設定。模型輸出五個 raw logits，推論時以 Sigmoid 轉為各類獨立機率，再依 validation-tuned thresholds 判定陽性標籤。〔待補正式文獻〕

## 2.9 大型語言模型輔助報告生成

本專題 Demo 中的 Ollama 只根據本機影像模型輸出的結構化結果產生說明文字，不直接判讀 X 光影像，也不接收 Ground Truth 作為生成內容。正式版本需補充大型語言模型輔助醫學報告生成之正式文獻。〔待補正式文獻〕

# 第三章　研究方法

## 3.1 整體研究架構

整體流程可分為 ROI 階段與 Full-image 階段。ROI 階段先以 BBox 標註建立五類 ROI 影像，再建立 balanced ROI dataset，並進行 RAD-DINO CLS 與 patch feature cache。其後將 feature cache 作為 teacher 訊號訓練 ConvNeXt-Tiny student，最後進行 ROI 五類單標籤分類比較。Full-image 階段則以 ROI Patch Proposed checkpoint 匯出的 backbone 作為初始化，訓練完整胸腔 X 光五類多標籤模型。

表 1　正式五類與任務定義

| 類別編號 | 英文類別名稱 | ROI 階段 | Full-image 階段 |
|---|---|---|---|
| 0 | Aortic enlargement | 單標籤分類候選類別 | 多標籤獨立陽性標籤 |
| 1 | Cardiomegaly | 單標籤分類候選類別 | 多標籤獨立陽性標籤 |
| 2 | Pleural thickening | 單標籤分類候選類別 | 多標籤獨立陽性標籤 |
| 3 | Pulmonary fibrosis | 單標籤分類候選類別 | 多標籤獨立陽性標籤 |
| 4 | Pleural effusion | 單標籤分類候選類別 | 多標籤獨立陽性標籤 |

## 3.2 資料來源與五類疾病

本研究的正式五類為 Aortic enlargement、Cardiomegaly、Pleural thickening、Pulmonary fibrosis 與 Pleural effusion，不包含 No Finding 類別。ROI 階段使用經 BBox 裁切後的單標籤資料；Full-image 階段則使用完整胸腔 X 光影像與五類多標籤 ground truth。

## 3.3 BBox 標註與 ROI 製作

ROI 製作流程依據 BBox 標註裁切病灶區域，並產生 224×224 ROI 版本供 feature cache、distillation 與分類訓練使用。現有 evidence 可確認曾進行 BBox overlay、crop 與 resize summary 層級人工檢查；逐筆人工審查紀錄目前未納入證據包，因此本文僅保守描述為 summary 層級品質檢查。

## 3.4 ROI 平衡與資料增強

ROI 原始資料共 4546 筆 ROI rows；balanced feature cache 共 4725 筆，五類各 945 筆。其中 4256 筆為原始 ROI，469 筆為可由 manifest 與檔名反向驗證之 brightness augmentation 影像。由於目前未找到 brightness generator 程式，本文不宣稱其產生方式為特定隨機公式。

表 2　ROI balanced dataset 類別數量

| 類別 | ROI rows | Brightness augmented rows |
|---|---|---|
| Aortic enlargement | 945 | 173 |
| Cardiomegaly | 945 | 162 |
| Pleural thickening | 945 | 0 |
| Pulmonary fibrosis | 945 | 0 |
| Pleural effusion | 945 | 134 |

表 3　Brightness augmentation factor 反向驗證分布

| Factor | 影像數量 |
|---|---|
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

## 3.5 Source-level 資料切分及資料洩漏控制

表 4　ROI Phase2 grouped split 類別數量

| Split | Total | Brightness rows | Aortic enlargement | Cardiomegaly | Pleural thickening | Pulmonary fibrosis | Pleural effusion |
|---|---|---|---|---|---|---|---|
| train | 3770 | 357 | 744 | 759 | 763 | 756 | 748 |
| val | 454 | 0 | 77 | 78 | 112 | 106 | 81 |
| test | 454 | 0 | 77 | 78 | 112 | 106 | 81 |

ROI Phase2 grouped split 中，train split 含 brightness augmentation；validation 與 test split 的 brightness augmented rows 皆為 0。此設定可避免以增強影像污染 validation 或 test 評估。

## 3.6 Phase 0：RAD-DINO CLS Feature Cache

RAD-DINO CLS teacher cache 使用 frozen/eval 模式，針對 ROI 224 balanced manifest 建立 teacher features，輸出形狀為 [4725, 768]。此階段僅進行 teacher feature extraction，不訓練 teacher。

## 3.7 Phase 0：RAD-DINO Patch Feature Cache

RAD-DINO Patch teacher cache 同樣使用 frozen/eval 模式。其處理器輸入為 [B, 3, 518, 518]，輸出 patch teacher features 形狀為 [4725, 768, 7, 7]，保留 7×7 空間格點表徵。

## 3.8 Phase 1：CLS Feature Distillation

CLS distillation 以 ConvNeXt-Tiny student 學習 RAD-DINO CLS feature。Loss 為 L2-normalized student feature 與 L2-normalized teacher feature 之 MSE；optimizer 為 AdamW；有效 batch size 為 64；訓練完成 30 epochs，checkpoint selection 依 best average distillation loss。

## 3.9 Phase 1：Patch Feature Distillation

Patch distillation 以 ConvNeXt-Tiny ImageNet1K V1 student 學習 RAD-DINO 7×7 patch feature。Loss 為沿 channel dimension 進行 L2 normalization 後的 float32 MSE；有效 batch size 為 64；訓練完成 84 epochs，並以 patch MSE/cosine 監控選擇最佳 checkpoint。

表 5　Phase 1 distillation 摘要

| 階段 | Completed epochs | Checkpoint selection | MSE / loss | Cosine similarity |
|---|---|---|---|---|
| CLS distillation | 30 | Best average distillation loss | 0.0003958256 | 0.8480030057 |
| Patch distillation | 84 | Best monitor patch MSE/cosine | 0.0002644237 | 0.89846133 |

## 3.10 Phase 2：ROI 五分類模型

ROI Phase2 為五類單標籤分類，輸出 5 logits，評估時使用 Softmax 機率。三個模型分別為 ImageNet Baseline、RAD-DINO CLS Proposed 與 RAD-DINO Patch Proposed。三者皆使用 CrossEntropyLoss、AdamW with CosineAnnealingLR、backbone learning rate 1e-05、head learning rate 0.0001、有效 batch size 64、最大 50 epochs，checkpoint selection 以 best validation loss 為準，且不使用 test set 選擇 checkpoint。

## 3.11 ROI 三模型比較設計

正式 ROI 比較採三模型公平比較資料夾中的輸出，不採早期二模型比較。公平性稽核顯示 augmentation 設定鎖定且 train-only，因此三模型比較具一致實驗條件。

## 3.12 ROI Patch Backbone 轉移

Full-image 階段的初始化來源為 ROI RAD-DINO Patch Proposed 匯出的 ConvNeXt-Tiny 五類 checkpoint。轉移時丟棄 ROI head，改接 Full-image 多標籤任務的新分類 head。此作法將 ROI patch-level distillation 學到的 backbone 表徵帶入完整影像任務。

## 3.13 Full-image 多標籤 Fine-tuning

表 6　Full-image 多標籤 split 與每類 positive count

| Split | Images | Aortic enlargement | Cardiomegaly | Pleural thickening | Pulmonary fibrosis | Pleural effusion |
|---|---|---|---|---|---|---|
| train | 472 | 280 | 279 | 280 | 278 | 280 |
| val | 59 | 35 | 36 | 35 | 36 | 35 |
| test | 59 | 35 | 35 | 35 | 36 | 35 |

Full-image 階段使用完整胸腔 X 光影像，不使用 BBox、ROI crop 或資料增強。輸入影像 resize 為 [3,224,224]；模型輸出五個 raw logits，訓練 loss 為 BCEWithLogitsLoss，推論時使用 Sigmoid 取得五類獨立機率。

## 3.14 Validation Threshold

表 7　Validation-tuned thresholds

| 類別 | Threshold | Validation F1 | Tie break |
|---|---|---|---|
| Aortic enlargement | 0.5 | 0.8493 | closest_to_0.5_then_lower_threshold |
| Cardiomegaly | 0.5 | 0.9333 | closest_to_0.5_then_lower_threshold |
| Pleural thickening | 0.39 | 0.8767 | closest_to_0.5_then_lower_threshold |
| Pulmonary fibrosis | 0.36 | 0.8000 | closest_to_0.5_then_lower_threshold |
| Pleural effusion | 0.34 | 0.8919 | closest_to_0.5_then_lower_threshold |

Threshold 搜尋範圍為 0.05 至 0.95，step 為 0.01；selection metric 為 per-class Validation F1。`validation_selected_thresholds.json` 顯示 `test_used` 為 False，因此 test set 未用於 threshold 選擇。

## 3.15 評估指標

ROI 單標籤分類使用 Accuracy、Macro-F1、Weighted-F1、Macro-AUROC 等指標。Full-image 多標籤分類使用 Macro-F1、Micro-F1、Exact Subset Accuracy、Hamming Loss、Macro-AUROC、Micro-AUROC 與各類 precision、recall、F1。Exact Subset Accuracy 表示一張影像所有標籤完全一致的比例，不能與 ROI 單標籤 Accuracy 直接等同。

## 3.16 實驗環境

本研究之正式訓練程式與輸出紀錄可追蹤至 evidence package；然而硬體型號、作業系統版本、CUDA 與 PyTorch 版本等完整環境資訊目前未在第二輪證據中統一整理，因此本小節保留為〔待確認〕。

# 第四章　實驗結果與分析

## 4.1 資料集統計

ROI balanced dataset 的五類數量一致，均為 945 筆。Full-image 多標籤資料則為 472 張 train、59 張 validation、59 張 test；因為多標籤任務允許同一張影像同時具有多個陽性類別，所以每類 positive count 的總和會大於影像張數。

## 4.2 CLS Distillation 結果

CLS distillation 訓練完成 30 epochs，最佳 average distillation loss 為 0.0003958256，最佳 average cosine similarity 為 0.8480030057。此結果顯示 student 可在 CLS feature 層級學習 teacher 表徵，但該階段並非直接分類結果。

## 4.3 Patch Distillation 結果

Patch distillation 訓練完成 84 epochs，Epoch 84 的 monitor MSE 為 0.0002644237，monitor cosine 為 0.89846133。相較 CLS distillation，patch feature 目標保留空間結構，可作為後續 ROI Patch Proposed 與 Full-image transfer 的基礎。

## 4.4 ROI 三模型整體結果

表 8　ROI 三模型正式 test 指標

| 模型 | Loss | Accuracy | Macro-F1 | Weighted-F1 | Macro-AUROC |
|---|---|---|---|---|---|
| ImageNet Baseline | 0.6040 | 0.7996 | 0.8080 | 0.7904 | 0.9484 |
| RAD-DINO CLS Proposed | 0.5315 | 0.7930 | 0.8017 | 0.7844 | 0.9507 |
| RAD-DINO Patch Proposed | 0.5214 | 0.7996 | 0.8053 | 0.7918 | 0.9506 |

三模型在 test Accuracy 上，ImageNet Baseline 與 RAD-DINO Patch Proposed 均為 0.7996；RAD-DINO CLS Proposed 為 0.7930。Macro-F1 最高為 ImageNet Baseline 的 0.8080，Patch Proposed 為 0.8053。Patch Proposed 在 Weighted-F1 與 Macro-AUROC 具競爭力，但不可描述為全面勝過 Baseline。

## 4.5 ROI 各類別結果

表 9　ROI per-class F1 比較

| 類別 | Baseline | CLS Proposed | Patch Proposed | Patch-Baseline |
|---|---|---|---|---|
| Aortic enlargement | 0.9872 | 0.9872 | 0.9872 | 0.0000 |
| Cardiomegaly | 0.9872 | 0.9935 | 0.9935 | 0.0064 |
| Pleural thickening | 0.5455 | 0.5417 | 0.6019 | 0.0564 |
| Pulmonary fibrosis | 0.8646 | 0.8621 | 0.8644 | -0.0002 |
| Pleural effusion | 0.6556 | 0.6243 | 0.5793 | -0.0762 |

Per-class F1 顯示 Patch Proposed 在 Pleural thickening 類別相對 Baseline 提升 0.0564，但在 Pleural effusion 類別低於 Baseline。此結果支持較細緻的疾病別討論，而不支持單一結論式宣稱。

## 4.6 Bootstrap 分析

表 10　ROI cluster bootstrap 95% CI 摘要

| 比較 | 指標 | Point diff | 95% CI | CI includes zero |
|---|---|---|---|---|
| cls_minus_baseline | accuracy | -0.0066 | [-0.030769927536, 0.015789473684] | True |
| cls_minus_baseline | macro_f1 | -0.0063 | [-0.030895667806, 0.015569776605] | True |
| cls_minus_baseline | weighted_f1 | -0.0060 | [-0.034069006804, 0.016931740776] | True |
| cls_minus_baseline | macro_auroc | 0.0022 | [-0.003352231600, 0.007290816853] | True |
| patch_minus_baseline | accuracy | 0.0000 | [-0.023986045537, 0.026727549316] | True |
| patch_minus_baseline | macro_f1 | -0.0027 | [-0.028549714353, 0.022868364608] | True |
| patch_minus_baseline | weighted_f1 | 0.0014 | [-0.024854160144, 0.028199338993] | True |
| patch_minus_baseline | macro_auroc | 0.0022 | [-0.006250717996, 0.010777456840] | True |
| patch_minus_cls | accuracy | 0.0066 | [-0.020179938397, 0.028283304746] | True |
| patch_minus_cls | macro_f1 | 0.0035 | [-0.025049546407, 0.027493963367] | True |
| patch_minus_cls | weighted_f1 | 0.0074 | [-0.023130412620, 0.034038887993] | True |
| patch_minus_cls | macro_auroc | -0.0001 | [-0.005262070189, 0.005070785840] | True |

Bootstrap 使用 source_image_id 作為 cluster unit，cluster count 為 59，seed 為 42，valid replicates 為 10000。所有主要比較的 95% CI 皆跨越 0，因此本研究不宣稱三模型間在主要指標上具有統計顯著差異。

## 4.7 Full-image 多標籤結果

表 11　Full-image 多標籤正式 test 指標

| 指標 | 數值 |
|---|---|
| Macro-F1 | 0.7865 |
| Micro-F1 | 0.7859 |
| Exact Subset Accuracy | 0.1695 |
| Hamming Loss | 0.2881 |
| Macro-AUROC | 0.7660 |
| Micro-AUROC | 0.7760 |

Full-image 多標籤模型於 test set 的 Macro-F1 為 0.7865、Micro-F1 為 0.7859。Exact Subset Accuracy 為 0.1695，表示所有五個標籤同時完全符合的比例。由於此任務允許多個陽性標籤，Exact Subset Accuracy 較嚴格，不應被當成一般單標籤 Accuracy 解讀。

## 4.8 Threshold 分析

Validation threshold tuning 顯示 Aortic enlargement 與 Cardiomegaly 使用 0.50；Pleural thickening、Pulmonary fibrosis 與 Pleural effusion 分別使用 0.39、0.36 與 0.34。較低 threshold 顯示模型需提高部分標籤的召回能力，但此選擇來自 validation set，並未使用 test set 調參。

## 4.9 模型結果討論

表 12　Full-image per-label test 指標

| 類別 | Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Aortic enlargement | 0.5 | 0.6591 | 0.8286 | 0.7342 | 29 | 15 | 6 |
| Cardiomegaly | 0.5 | 0.8056 | 0.8286 | 0.8169 | 29 | 7 | 6 |
| Pleural thickening | 0.39 | 0.6977 | 0.8571 | 0.7692 | 30 | 13 | 5 |
| Pulmonary fibrosis | 0.36 | 0.6538 | 0.9444 | 0.7727 | 34 | 18 | 2 |
| Pleural effusion | 0.34 | 0.7391 | 0.9714 | 0.8395 | 34 | 12 | 1 |

Full-image per-label 結果顯示 Cardiomegaly 與 Pleural effusion 的 F1 較高，Aortic enlargement 相對受 FP 影響，Pulmonary fibrosis 則具有高 recall 但 precision 較低。此結果較適合解讀為不同標籤間錯誤型態不同，而非單純以一個總分代表全部疾病。

## 4.10 研究限制

本研究目前仍有四項重要限制：brightness augmentation generator 未找到、人工審查僅具 summary 層級證據、尚無外部資料集泛化驗證、Full-image multilabel 尚未建立 bootstrap 信賴區間。這些限制不影響保守描述已驗證結果，但會限制統計顯著性、外部泛化與臨床可用性的主張。

# 第五章　系統設計與實作

## 5.1 系統需求

最終系統需求為在本機環境中提供 Full-image 五類多標籤胸腔 X 光推論 Demo，支援影像上傳、五類機率顯示、Ground Truth 比對、Ollama 輔助說明、Markdown/HTML/PDF 報告輸出與錯誤處理。系統不依賴外部雲端服務才能完成展示。

## 5.2 系統整體架構

系統由 Gradio UI、Full-image multilabel inference service、Ollama service、report prompt/report rendering 與 session output 組成。模型推論服務負責影像前處理、checkpoint 載入、Sigmoid 機率與 threshold 判定；Ollama service 僅接收本機模型結構化推論結果並產生保守說明。

## 5.3 Gradio 使用者介面

Gradio Demo 提供病人資訊欄位、完整胸腔 X 光影像上傳、開始分析、重新產生說明、列印/匯出 PDF、清除按鈕、五類模型機率、Ground Truth 比對、模型預測摘要、AI 輔助診斷說明書與系統狀態。病人資訊僅作 Demo 報告顯示，不參與模型推論。

## 5.4 Full-image 推論服務

推論服務載入正式 Full-image ConvNeXt-Tiny multilabel checkpoint 與 validation thresholds，對單張完整胸腔 X 光影像進行前處理與推論。輸出包含五類機率、依 threshold 判定的 predicted positive labels、模型摘要與後續報告所需之結構化欄位。

## 5.5 影像前處理

Full-image inference 使用完整影像，不執行 BBox 偵測或 ROI crop。依 evidence，正式 Full-image training config 中 augmentation 為 false；Demo 推論亦不應被描述為 ROI 或 YOLO 流程。

## 5.6 Sigmoid 與 Threshold 判定

Full-image 模型輸出五個 raw logits，經 Sigmoid 轉換為每一類的獨立機率。每類再與 validation-selected threshold 比較，以取得多標籤陽性預測。此流程不同於 ROI 階段的 Softmax 單標籤推論。

## 5.7 Ground Truth 查詢與隔離

Ground Truth 僅在模型推論完成後用於 Demo 驗證與 TP、FP、FN 比對，不輸入模型，也不傳送給 Ollama。此隔離可避免把標註答案洩漏至模型或文字生成流程。

## 5.8 Ollama 輔助說明

Ollama 在系統中負責將模型輸出的類別、機率與信心程度轉換為繁體中文白話說明。Ollama 不直接讀取或判讀胸腔 X 光影像，因此其輸出不得被寫成正式醫療診斷。

## 5.9 Session 與稽核紀錄

系統可針對單次 Demo 推論建立 session output，保存模型推論結果、報告與必要稽核欄位。論文僅描述系統功能，不將大量影像或可識別病患資訊插入文件。

## 5.10 CSV、JSON、Markdown 與 PDF 輸出

Demo 支援將單張影像推論報告輸出為 Markdown、HTML 與 PDF 類型文件。報告固定包含非正式診斷警語，並以保守語氣呈現模型預測。

## 5.11 錯誤處理與系統穩定性

系統設計包含模型載入、影像處理、Ollama 呼叫與報告生成的錯誤處理。若 Ollama 不可用，系統可降級為規則式說明，以確保 Demo 不依賴外部服務或單一生成流程。

## 5.12 系統限制

本系統用途為學術研究展示與輔助理解，不是臨床醫療器材，也不能取代醫師判讀。Demo 中顯示的 Ground Truth 僅供驗證，Ollama 產生之文字僅根據模型輸出進行說明。

# 第六章　結論與未來工作

## 6.1 研究成果總結

本研究完成由 ROI 五類單標籤分類、RAD-DINO feature distillation、ConvNeXt-Tiny 三模型比較，到 Full-image 五類多標籤模型與 Gradio Demo 的完整研究流程。正式證據顯示 ROI 三模型表現相近，Patch Proposed 在部分疾病別指標具競爭力，但統計檢定未支持主要指標具有顯著差異。Full-image 模型則完成五類多標籤 test evaluation，並整合為可展示的本機 Demo。

## 6.2 研究貢獻

- 建立可追蹤的 ROI balanced dataset 與 source-level grouped split。
- 比較 RAD-DINO CLS 與 patch feature distillation 對 ConvNeXt-Tiny downstream classification 的影響。
- 將 ROI Patch Proposed backbone 轉移至 Full-image multilabel 任務，形成最終 Demo 推論模型。
- 在 Demo 中明確分離模型推論、Ground Truth 驗證與 Ollama 文字說明，降低錯誤解讀風險。

## 6.3 研究限制

本研究仍受限於 evidence package 中可驗證資料。Brightness augmentation 產生器尚未定位；人工審查缺少逐筆紀錄；尚無外部資料集驗證；Full-image multilabel 尚未建立 bootstrap CI。這些限制使本文不宣稱外部泛化、臨床效益或統計顯著優勢。

## 6.4 未來工作

- 補充 brightness augmentation generator 或將其限制明確寫入附錄。
- 建立完整逐筆人工審查紀錄與 failure type 統計。
- 加入外部資料集評估，以確認跨來源影像泛化能力。
- 對 Full-image multilabel 指標補做 bootstrap 信賴區間。
- 進行 calibration、使用者研究與醫療專業人員評估，但不得在未完成前宣稱臨床可用。

# 附錄 A　圖表占位與後續補圖

基於隱私保護，本初稿未插入完整 Dataset 或大量胸腔 X 光影像。後續若需加入 Demo 截圖或範例影像，應使用匿名資料並遮蔽病人資訊。

圖 A-1　〔待補：整體研究流程圖〕

圖 A-2　〔待補：ROI confusion matrices〕

圖 A-3　〔待補：ROI per-class F1 bar chart〕

圖 A-4　〔待補：Full-image per-label metrics bar chart〕

圖 A-5　〔待補：Gradio Demo 匿名截圖〕

圖 A-6　〔待補：Session output 匿名範例〕
