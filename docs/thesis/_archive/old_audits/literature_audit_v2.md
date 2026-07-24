# Literature Audit v2

## Scope

- Reviewed target: Chapter 2 of `docs/thesis/draft/胸腔X光影像分類與知識蒸餾系統_論文修正版_v2.docx`.
- This file only maps missing formal literature support. The DOCX was not modified.
- Evidence inputs: `full_thesis_consistency_audit_v2.md`, `full_thesis_corrections_v2.csv`, `thesis_evidence_trace_v2.csv`, and direct Chapter 2 DOCX text extraction.
- Citation numbering below is a proposed TANET-style first-citation order for a future v3 revision.

## Placeholder Summary

- `full_thesis_corrections_v2.csv` records one Chapter 2 missing-citation issue covering nine formal-literature placeholders.
- Direct Chapter 2 text inspection identifies placeholders in sections 2.1 through 2.9.
- All nine placeholders can be supported by formal literature, with project-specific implementation statements still requiring the internal evidence trace.

## Placeholder Mapping

### P01 - 2.1 胸腔 X 光影像分析

- Original paragraph summary: 胸腔 X 光影像低成本、低輻射且臨床可近，但判讀受解剖重疊、拍攝姿勢、病灶大小與影像品質影響；需補自動化輔助判讀文獻。
- Claim needing support: 胸腔 X 光為常見且可近的影像檢查；自動化胸腔 X 光分析常以多標籤分類/定位資料集作為研究基礎。
- Recommended citation(s): [1], [2], [3]
- Why these sources fit: Broder 可支持胸腔 X 光檢查特性與臨床使用脈絡；ChestX-ray8 與 CheXpert 可支持胸腔 X 光自動化、多標籤與 benchmark 研究背景。
- Formal source(s):
  - [1] Joshua Broder, *Imaging the Chest: The Chest Radiograph*, 2011; DOI: 10.1016/B978-1-4160-6113-7.10005-5. Source: https://scholars.duke.edu/publication/964986
  - [2] Xiaosong Wang; Yifan Peng; Le Lu; Zhiyong Lu; Mohammadhadi Bagheri; Ronald M. Summers, *ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, 2017; DOI: not listed in verified source. Source: https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html
  - [3] Jeremy Irvin et al., *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*, 2019; DOI: 10.1609/aaai.v33i01.3301590. Source: https://ojs.aaai.org/index.php/AAAI/article/view/3834
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: 若正文要描述特定臨床適應症，需另補專門臨床指引；目前第二章技術背景可安全引用。
- Safe thesis wording: 胸腔 X 光為臨床常見且相對容易取得的影像檢查；既有大型資料集研究也將其作為自動化胸腔疾病標籤分類與定位任務的主要影像來源 [1]-[3]。

### P02 - 2.2 深度學習影像分類

- Original paragraph summary: 深度學習可自動學習階層式影像特徵；卷積神經網路常用於醫學影像分類、偵測與分割。
- Claim needing support: 深度學習與卷積神經網路已廣泛用於醫學影像分析，並可從資料中學習階層式表徵。
- Recommended citation(s): [4], [5]
- Why these sources fit: 兩篇綜述皆直接討論深度學習在醫學影像分類、偵測、分割與特徵學習上的應用。
- Formal source(s):
  - [4] Geert Litjens et al., *A survey on deep learning in medical image analysis*, 2017; DOI: 10.1016/j.media.2017.07.005. Source: https://pubmed.ncbi.nlm.nih.gov/28778026/
  - [5] Dinggang Shen; Guorong Wu; Heung-Il Suk, *Deep Learning in Medical Image Analysis*, 2017; DOI: 10.1146/annurev-bioeng-071516-044442. Source: https://doi.org/10.1146/ANNUREV-BIOENG-071516-044442
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: 本段若要提 ConvNeXt，建議移至 2.5 或另以 ConvNeXt 原論文支持。
- Safe thesis wording: 深度學習，尤其是卷積神經網路，已被廣泛應用於醫學影像分類、偵測與分割，並能由資料中學習階層式影像表徵 [4], [5]。

### P03 - 2.3 Vision Transformer 與自監督學習

- Original paragraph summary: 自監督學習可利用未標註影像學習一般化特徵；Vision Transformer 以 patch 為單位建模影像；本研究使用 RAD-DINO 作 frozen teacher。
- Claim needing support: ViT 以影像 patch 序列作為輸入；DINOv2 類自監督方法可學習可轉移視覺特徵；RAD-DINO 是醫學影像 encoder。
- Recommended citation(s): [6], [7], [8]
- Why these sources fit: ViT 原文支持 patch-based Transformer；DINOv2 原文支持自監督視覺特徵；RAD-DINO 原文支持醫學影像 encoder 背景。
- Formal source(s):
  - [6] Alexey Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2021; DOI: 10.48550/arXiv.2010.11929. Source: https://arxiv.org/abs/2010.11929
  - [7] Maxime Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2024; DOI: 10.48550/arXiv.2304.07193. Source: https://arxiv.org/abs/2304.07193
  - [8] Fernando Pérez-García et al., *Exploring scalable medical image encoders beyond text supervision*, 2025; DOI: 10.1038/s42256-024-00965-w. Source: https://www.nature.com/articles/s42256-024-00965-w
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: RAD-DINO 在本專題中作 frozen teacher 且不是最終 Demo 推論模型，屬專案事實，應同時由 evidence trace 支持。
- Safe thesis wording: Vision Transformer 將影像切分為 patch 序列進行建模；自監督視覺模型如 DINOv2 旨在學習可轉移的視覺特徵，而 RAD-DINO 則將相關概念延伸至醫學影像 encoder [6]-[8]。

### P04 - 2.4 RAD-DINO 醫學影像基礎模型

- Original paragraph summary: RAD-DINO 在本研究中作為醫學影像特徵 teacher，提供 CLS feature 與 patch feature 作為蒸餾目標。
- Claim needing support: RAD-DINO 是以醫學影像為主的可擴展 image encoder；可作為本研究 feature teacher 的相關方法依據。
- Recommended citation(s): [8]
- Why these sources fit: Nature Machine Intelligence 論文正式介紹 RAD-DINO 的醫學影像 encoder 定位、訓練方向與下游任務評估。
- Formal source(s):
  - [8] Fernando Pérez-García et al., *Exploring scalable medical image encoders beyond text supervision*, 2025; DOI: 10.1038/s42256-024-00965-w. Source: https://www.nature.com/articles/s42256-024-00965-w
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: CLS/patch feature 的實際 tensor 形狀與本專題使用方式須引用 evidence package，而不是由 RAD-DINO 論文推導。
- Safe thesis wording: RAD-DINO 是針對醫學影像表徵學習提出的 biomedical image encoder；本研究將其固定作為 teacher，以其影像層級與 patch-level feature 作為蒸餾目標 [8]。

### P05 - 2.5 ConvNeXt

- Original paragraph summary: ConvNeXt-Tiny 為本專題 student backbone 與後續分類模型基礎，可接收蒸餾後表徵並用於 ROI 或 Full-image 任務。
- Claim needing support: ConvNeXt 是現代化卷積神經網路家族；本研究採用 ConvNeXt-Tiny 作 student/backbone 屬專案設計。
- Recommended citation(s): [9]
- Why these sources fit: ConvNeXt 原論文提出 modernized ConvNet family，支持 ConvNeXt 作為現代 CNN backbone 的背景敘述。
- Formal source(s):
  - [9] Zhuang Liu; Hanzi Mao; Chao-Yuan Wu; Christoph Feichtenhofer; Trevor Darrell; Saining Xie, *A ConvNet for the 2020s*, 2022; DOI: not listed in verified source. Source: https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: ConvNeXt-Tiny 具體用於 student 與 Full-image 初始化的事實應以專案訓練設定與 evidence trace 支持。
- Safe thesis wording: ConvNeXt 重新檢視並現代化卷積網路設計，提出可作為通用視覺 backbone 的 ConvNet family；本研究採用其中 Tiny 規模作為 student 與後續分類 backbone [9]。

### P06 - 2.6 知識蒸餾

- Original paragraph summary: 知識蒸餾讓 student 學習 teacher 輸出的表徵或預測；本研究採 feature-level distillation。
- Claim needing support: 知識蒸餾是將大型或集成模型知識轉移到較小 student 模型的訓練方法；本研究為 feature-level distillation。
- Recommended citation(s): [10]
- Why these sources fit: Hinton 等人的原始蒸餾論文支持 teacher-student 知識轉移概念；feature-level 具體設計則由本研究方法與訓練設定支持。
- Formal source(s):
  - [10] Geoffrey Hinton; Oriol Vinyals; Jeff Dean, *Distilling the Knowledge in a Neural Network*, 2015; DOI: 10.48550/arXiv.1503.02531. Source: https://arxiv.org/abs/1503.02531
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: 本研究的 CLS/Patch feature-level loss 不應寫成 Hinton 原文完全相同方法；需說明是採其 teacher-student 思想的特徵層蒸餾變形。
- Safe thesis wording: 知識蒸餾的核心概念是將 teacher 模型或模型群的知識轉移至較輕量 student；本研究延伸此概念，採用 feature-level 對齊進行 CLS 與 patch feature 蒸餾 [10]。

### P07 - 2.7 CLS 與 Patch 特徵

- Original paragraph summary: CLS feature 表示影像層級摘要；patch feature 保留空間網格資訊；比較兩者對 ROI 下游分類的影響。
- Claim needing support: ViT 類模型使用 class-level 與 patch-level representations；DINOv2/RAD-DINO 提供可用於下游任務的視覺特徵。
- Recommended citation(s): [6], [7], [8]
- Why these sources fit: ViT 支持 patch 序列與 class-level representation 背景；DINOv2 與 RAD-DINO 支持自監督/醫學影像特徵作為下游表徵的脈絡。
- Formal source(s):
  - [6] Alexey Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2021; DOI: 10.48550/arXiv.2010.11929. Source: https://arxiv.org/abs/2010.11929
  - [7] Maxime Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2024; DOI: 10.48550/arXiv.2304.07193. Source: https://arxiv.org/abs/2304.07193
  - [8] Fernando Pérez-García et al., *Exploring scalable medical image encoders beyond text supervision*, 2025; DOI: 10.1038/s42256-024-00965-w. Source: https://www.nature.com/articles/s42256-024-00965-w
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: Patch Proposed backbone 轉移至 Full-image 是本專題實驗設計，需引用 evidence package，不可只由外部文獻支持。
- Safe thesis wording: 在 ViT 類架構中，影像可被表示為多個 patch-level feature，並可透過 class-level feature 形成影像摘要；本研究據此比較 CLS 與 patch feature 蒸餾對下游 ROI 分類的影響 [6]-[8]。

### P08 - 2.8 多標籤分類

- Original paragraph summary: Full-image 階段一張影像可能同時包含多個目標標籤；輸出 raw logits，推論時用 Sigmoid 與 validation-tuned thresholds。
- Claim needing support: 胸腔 X 光資料常可包含多個影像標籤；多標籤分類通常以各類別獨立機率或 threshold 進行判定。本研究 Sigmoid/threshold 是內部實作。
- Recommended citation(s): [2], [3]
- Why these sources fit: ChestX-ray8 與 CheXpert 都明確以胸腔 X 光多標籤/多觀察標籤為研究設定，支持 Full-image 多標籤背景。
- Formal source(s):
  - [2] Xiaosong Wang; Yifan Peng; Le Lu; Zhiyong Lu; Mohammadhadi Bagheri; Ronald M. Summers, *ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, 2017; DOI: not listed in verified source. Source: https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html
  - [3] Jeremy Irvin et al., *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*, 2019; DOI: 10.1609/aaai.v33i01.3301590. Source: https://ojs.aaai.org/index.php/AAAI/article/view/3834
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: Validation-tuned thresholds 的數值與來源必須引用本專題 threshold JSON 與 evidence trace。
- Safe thesis wording: 胸腔 X 光公開資料集研究常將一張影像對應到多個可能觀察或疾病標籤，因此 Full-image 階段採多標籤分類較符合多病灶共存情境 [2], [3]。

### P09 - 2.9 大型語言模型輔助報告生成

- Original paragraph summary: Demo 中 Ollama 只根據本機模型輸出的結構化結果產生說明文字，不直接判讀 X 光，也不接收 Ground Truth。
- Claim needing support: 放射報告自動生成與醫療語言模型已有相關研究，但本專題僅將 LLM 用於本機模型輸出的保守文字說明。
- Recommended citation(s): [11], [12]
- Why these sources fit: R2Gen 支持放射報告生成作為醫學影像 AI 任務；Singhal 等人支持大型語言模型在醫療知識文字任務中的研究脈絡與需謹慎評估的立場。
- Formal source(s):
  - [11] Zhihong Chen; Yan Song; Tsung-Hui Chang; Xiang Wan, *Generating Radiology Reports via Memory-driven Transformer*, 2020; DOI: 10.18653/v1/2020.emnlp-main.112. Source: https://aclanthology.org/2020.emnlp-main.112/
  - [12] Karan Singhal et al., *Large language models encode clinical knowledge*, 2023; DOI: 10.1038/s41586-023-06291-2. Source: https://www.nature.com/articles/s41586-023-06291-2
- Verification status: verified
- Safe to add to thesis: yes
- Remaining manual check: Ollama 不直接判讀影像、不接收 Ground Truth，屬本專題系統設計事實，需以 demo execution flow 與程式證據支持。
- Safe thesis wording: 既有研究已探討放射報告自動生成與醫療大型語言模型的知識能力；本研究僅將本機 LLM 作為結構化模型輸出的文字化輔助說明器，並避免將其描述為影像分類模型或正式診斷工具 [11], [12]。

## Proposed Reference List

- [1] Joshua Broder, "Imaging the Chest: The Chest Radiograph," Diagnostic Imaging for the Emergency Physician, pp. 185-296, 2011. doi:10.1016/B978-1-4160-6113-7.10005-5. Source: https://scholars.duke.edu/publication/964986
- [2] Xiaosong Wang; Yifan Peng; Le Lu; Zhiyong Lu; Mohammadhadi Bagheri; Ronald M. Summers, "ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2097-2106, 2017. Source: https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html
- [3] Jeremy Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 01, pp. 590-597, 2019. doi:10.1609/aaai.v33i01.3301590. Source: https://ojs.aaai.org/index.php/AAAI/article/view/3834
- [4] Geert Litjens et al., "A survey on deep learning in medical image analysis," Medical Image Analysis, vol. 42, pp. 60-88, 2017. doi:10.1016/j.media.2017.07.005. Source: https://pubmed.ncbi.nlm.nih.gov/28778026/
- [5] Dinggang Shen; Guorong Wu; Heung-Il Suk, "Deep Learning in Medical Image Analysis," Annual Review of Biomedical Engineering, vol. 19, pp. 221-248, 2017. doi:10.1146/annurev-bioeng-071516-044442. Source: https://doi.org/10.1146/ANNUREV-BIOENG-071516-044442
- [6] Alexey Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," International Conference on Learning Representations (ICLR), 2021. doi:10.48550/arXiv.2010.11929. Source: https://arxiv.org/abs/2010.11929
- [7] Maxime Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," Transactions on Machine Learning Research / arXiv, 2024. doi:10.48550/arXiv.2304.07193. Source: https://arxiv.org/abs/2304.07193
- [8] Fernando Pérez-García et al., "Exploring scalable medical image encoders beyond text supervision," Nature Machine Intelligence, vol. 7, pp. 119-130, 2025. doi:10.1038/s42256-024-00965-w. Source: https://www.nature.com/articles/s42256-024-00965-w
- [9] Zhuang Liu; Hanzi Mao; Chao-Yuan Wu; Christoph Feichtenhofer; Trevor Darrell; Saining Xie, "A ConvNet for the 2020s," IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976-11986, 2022. Source: https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html
- [10] Geoffrey Hinton; Oriol Vinyals; Jeff Dean, "Distilling the Knowledge in a Neural Network," NIPS Deep Learning and Representation Learning Workshop / arXiv, 2015. doi:10.48550/arXiv.1503.02531. Source: https://arxiv.org/abs/1503.02531
- [11] Zhihong Chen; Yan Song; Tsung-Hui Chang; Xiang Wan, "Generating Radiology Reports via Memory-driven Transformer," Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1439-1449, 2020. doi:10.18653/v1/2020.emnlp-main.112. Source: https://aclanthology.org/2020.emnlp-main.112/
- [12] Karan Singhal et al., "Large language models encode clinical knowledge," Nature, vol. 620, no. 7972, pp. 172-180, 2023. doi:10.1038/s41586-023-06291-2. Source: https://www.nature.com/articles/s41586-023-06291-2

## Resolution Notes

- Found literature support for all nine Chapter 2 placeholders.
- Recommended 12 distinct formal references.
- Ten references have verified DOI values; two CVF conference papers were verified through official CVF pages without DOI fields in the checked source.
- Three references use official arXiv pages as the verified source; each has an arXiv-issued DOI.
- No Wikipedia, blog, generic tutorial, or automatically generated summary source was selected.
- Project-specific claims, such as the exact role of RAD-DINO, ConvNeXt-Tiny, thresholds, Ground Truth isolation, and Ollama flow, should remain tied to `thesis_evidence_trace_v2.csv` and the evidence package rather than external literature alone.
- The future v3 DOCX can safely replace each Chapter 2 placeholder with the suggested citations if the user approves the wording pass.
