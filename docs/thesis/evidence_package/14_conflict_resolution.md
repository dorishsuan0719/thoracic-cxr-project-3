# 14 Conflict Resolution

本文件為第二輪衝突解決紀錄。查證範圍包含 `08_augmentation_audit.md`、`10_conflicts_and_unknowns.md`、`11_verified_thesis_facts.md`，以及其引用的原始程式、CSV、JSON、audit、metrics 與 manifest。本輪不修改任何程式、資料、checkpoint、threshold、metrics 或既有 evidence 文件，只整理可採信事實與仍須保守標記的項目。

## 總結

| 項目 | 第二輪結論 |
|---|---|
| 5 項 conflicts | 4 項可解決，1 項部分解決但保留待確認 |
| 4 項待確認 | 0 項完全解決，1 項補充高層級證據但仍需使用者確認細節 |
| 可直接寫入論文 | 正式資料量、正式模型、正式指標、augmentation 實際使用階段、Demo 流程 |
| 不可寫成定論 | brightness 產生器實作、人工審查逐筆結果、外部泛化、Full-image bootstrap 顯著性 |

## Conflict 1：舊文件亂碼或過期文字與正式輸出衝突

### 1. 問題內容

部分舊版 Markdown 或筆記內容存在亂碼、過期描述或未經正式輸出驗證的文字，可能與正式 CSV、JSON、audit、metrics 結果不一致。

### 2. 衝突或缺少證據的檔案

- `docs/final_patch_distillation_project/final_research_report.md`
- `docs/final_patch_distillation_project/project_final_status.md`
- `docs/final_patch_distillation_project/presentation_outline.md`
- 正式來源：
  - `outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv`
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/tables/overall_metrics_comparison.csv`
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/statistics/cluster_bootstrap_results.csv`
  - `outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/test_metrics.json`

### 3. 原始查證

回查正式 manifest、metrics、audit 與 training config 後，正式數字可由 CSV/JSON 直接支持；舊版 Markdown 只能作為敘事草稿，不應作為數字來源。

### 4. 是否可以解決

可以解決。

### 5. 正式採用事實與來源

- ROI balanced dataset 為 4,725 張、五類各 945 張：`outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv`。
- ROI 三模型正式比較：`outputs/raddino_convnext_tiny_three_model_comparison_seed42/tables/overall_metrics_comparison.csv`。
- Bootstrap 結果：`outputs/raddino_convnext_tiny_three_model_comparison_seed42/statistics/cluster_bootstrap_results.csv` 與 `research_conclusion.md`。
- Full-image 正式 test metrics：`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/test_metrics.json`。

### 6. 未解決內容

無。舊版 Markdown 的敘事不採為主要證據。

### 7. 影響章節

影響「實驗設定」、「實驗結果」與「討論」章節中的數字引用方式。

### 8. 論文安全寫法

「本研究之資料量、模型表現與統計檢定結果，以正式輸出的 CSV、JSON、audit 與 metrics 檔案為依據；舊版筆記僅作為流程參考，不作為定量結果來源。」

## Conflict 2：正式 ROI 模型比較來源不一致

### 1. 問題內容

專案內同時存在早期二模型比較與後續三模型比較資料夾，容易誤把早期結果當成正式論文結果。

### 2. 衝突或缺少證據的檔案

- 早期比較：
  - `outputs/raddino_convnext_tiny_experiment_seed42/final_comparison/final_comparison_report.md`
- 正式三模型比較：
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/comparison_summary.json`
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/tables/overall_metrics_comparison.csv`
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/statistics/cluster_bootstrap_results.csv`
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/research_conclusion.md`
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/fairness_audit.json`

### 3. 原始查證

正式三模型比較包含 ImageNet Baseline、RAD-DINO CLS Proposed、RAD-DINO Patch Proposed，並有公平性稽核與 bootstrap 統計。早期二模型比較不包含完整正式比較設計。

### 4. 是否可以解決

可以解決。

### 5. 正式採用事實與來源

正式 ROI 比較只採用 `outputs/raddino_convnext_tiny_three_model_comparison_seed42/`。該資料夾中 `fairness_audit.json` 顯示 augmentation 設定鎖定且 train-only；`cluster_bootstrap_results.csv` 顯示主要指標的 95% CI 均跨越 0。

### 6. 未解決內容

無。

### 7. 影響章節

影響「實驗設計」、「結果比較」、「統計分析」。

### 8. 論文安全寫法

「ROI 階段正式比較採用三模型公平比較資料夾之結果；早期二模型比較僅作為開發歷程，不納入正式主結果。」

## Conflict 3：舊專案路徑引用是否影響正式 project-3

### 1. 問題內容

專案中曾出現舊專案路徑引用，需確認正式 Demo 與正式資源是否仍依賴 `thoracic-cxr-project`、`thoracic-cxr-yolo-project` 或 `thoracic-cxr-project_OLD_DISABLED`。

### 2. 衝突或缺少證據的檔案

- 路徑稽核輸出：
  - `outputs/path_audit/old_path_references.csv`
  - `outputs/path_audit/project3_independence_summary.json`
- 正式 Demo 與服務：
  - `app_full_image_multilabel_ollama_gradio.py`
  - `src/full_image_multilabel_inference_service.py`
  - `src/full_image_multilabel_ollama_service.py`
  - `src/full_image_multilabel_report_prompt.py`

### 3. 原始查證

正式主程式支援 `--project-root`，正式服務以 project root 解析 checkpoint、threshold、Ground Truth catalog 與 session output。舊路徑引用主要保留於路徑稽核紀錄或歷史文字，不作為正式 runtime 依賴。

### 4. 是否可以解決

可以解決。

### 5. 正式採用事實與來源

- 正式主程式：`app_full_image_multilabel_ollama_gradio.py`。
- 正式後端：`src/full_image_multilabel_inference_service.py`、`src/full_image_multilabel_ollama_service.py`、`src/full_image_multilabel_report_prompt.py`。
- 正式 checkpoint 與 threshold 由 project-3 內相對路徑解析。

### 6. 未解決內容

無。論文不需引用舊路徑。

### 7. 影響章節

影響「系統實作」與「Demo 部署」。

### 8. 論文安全寫法

「最終 Demo 已整理為 project-3 內可獨立執行的 Gradio 系統；推論服務以本專案內正式 checkpoint、threshold 與 Ground Truth catalog 執行。」

## Conflict 4：469 張 brightness augmentation 的來源與產生公式

### 1. 問題內容

balanced ROI manifest 中存在 469 張 brightness augmentation 影像，檔名或 manifest 可反向驗證 brightness factor，但目前未找到產生這些檔案的 generator 程式。因此不可把輸出檔案的 factor 範圍直接推論成 `random.uniform(0.95, 1.05)`。

### 2. 衝突或缺少證據的檔案

- 可驗證輸出：
  - `outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv`
  - `outputs/raddino_convnext_tiny_experiment_seed42/phase2_split/train_roi_manifest.csv`
  - `outputs/raddino_convnext_tiny_experiment_seed42/phase2_split/val_roi_manifest.csv`
  - `outputs/raddino_convnext_tiny_experiment_seed42/phase2_split/test_roi_manifest.csv`
- 解析與驗證程式：
  - `src/audit_balanced_roi_and_build_manifest.py`
  - `src/create_phase2_grouped_split.py`
- 缺少證據：
  - 實際建立 brightness augmentation 影像的 generator 程式或執行紀錄。

### 3. 原始查證

`roi_manifest.csv` 共 4,725 rows，其中 469 rows 標示為 brightness augmentation，檔名 factor 分布如下：

| brightness factor | 影像數量 |
|---:|---:|
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

總數為 469，範圍為 0.95 到 1.05。Phase2 train split 中含 357 張 brightness augmentation；val 與 test split 中為 0 張。

### 4. 是否可以解決

部分解決。可確認「輸出檔案中實際存在的 factor 值、數量與 split 使用情形」；不可確認「產生器使用的隨機公式或抽樣策略」。

### 5. 正式採用事實與來源

- `roi_manifest.csv` 中可反向驗證 469 張 brightness augmentation，factor 實際值為 0.95 至 1.05。
- `train_roi_manifest.csv` 中 brightness augmentation 僅進入 train；`val_roi_manifest.csv` 與 `test_roi_manifest.csv` 中 augmented count 為 0。
- `src/audit_balanced_roi_and_build_manifest.py` 的 `parse_filename`、`is_brightness_augmented` 用於解析檔名與標記 brightness augmentation。
- `src/create_phase2_grouped_split.py` 驗證 balanced manifest 中 augmented count 與 val/test 不含 augmented rows。

### 6. 未解決內容

仍保留〔待確認〕：brightness augmentation 影像的產生器位置、抽樣策略，以及是否以固定清單或隨機程序建立。

### 7. 影響章節

影響「資料前處理與資料增強」章節。若未補證據，不可描述成隨機 brightness augmentation，只能描述成 manifest 反向驗證到的離線 brightness 補充資料。

### 8. 論文安全寫法

「在 balanced ROI manifest 中可觀察到 469 張以檔名標記為 brightness augmentation 的影像，其 factor 值介於 0.95 至 1.05；本研究僅能由輸出檔案反向驗證該事實，未在專案中找到可確認其產生抽樣公式的程式。」

## Conflict 5：正式指標、Validation/Test 與 Accuracy 定義混用

### 1. 問題內容

ROI 階段為五類單標籤分類，Full-image 階段為五類多標籤分類。ROI Accuracy 與 Full-image Exact Subset Accuracy 含義不同；validation threshold selection 與 test evaluation 也不可混用。

### 2. 衝突或缺少證據的檔案

- ROI：
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/tables/overall_metrics_comparison.csv`
  - `outputs/raddino_convnext_tiny_three_model_comparison_seed42/statistics/cluster_bootstrap_results.csv`
- Full-image：
  - `outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/validation_selected_thresholds.json`
  - `outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/test_metrics.json`
  - `outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/per_class_test_metrics.csv`

### 3. 原始查證

ROI formal comparison 報告單標籤 Accuracy、Macro-F1、Weighted-F1、Macro-AUROC。Full-image 報告 multilabel Macro-F1、Micro-F1、Exact Subset Accuracy、Hamming Loss、AUROC 與 per-class metrics。Full-image threshold 來源為 validation split，test metrics 為單次 test evaluation。

### 4. 是否可以解決

可以解決。

### 5. 正式採用事實與來源

- ROI ImageNet Baseline test Accuracy = 0.799559471366，Macro-F1 = 0.807999579266。
- ROI RAD-DINO CLS Proposed test Accuracy = 0.792951541850，Macro-F1 = 0.801748192622。
- ROI RAD-DINO Patch Proposed test Accuracy = 0.799559471366，Macro-F1 = 0.805259370123。
- Full-image test Macro-F1 = 0.7865085676876251，Micro-F1 = 0.7858942065491183，Exact Subset Accuracy = 0.1694915254237288。
- Bootstrap 主要指標 95% CI 均跨越 0，不支持「Patch Proposed 具有統計顯著優勢」。

### 6. 未解決內容

Full-image multilabel 尚未看到 bootstrap CI 檔案；若要做顯著性主張需另外補證據。

### 7. 影響章節

影響「實驗結果」、「統計檢定」、「討論」。

### 8. 論文安全寫法

「ROI 與 Full-image 任務之指標不可直接以 Accuracy 對等比較；本研究分別回報 ROI 單標籤分類指標與 Full-image 多標籤指標。Bootstrap 結果未支持主要模型差異達統計顯著。」

## A. 469 張 brightness augmentation 再確認

正式可寫事實如下：

- 總數：469 張。
- 來源：`outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv`。
- 實際 factor：0.95、0.96、0.97、0.98、0.99、1.01、1.02、1.03、1.04、1.05。
- 範圍：0.95 到 1.05。
- 不包含 1.00 factor。
- 不可宣稱找到 `random.uniform(0.95, 1.05)` 產生器。
- 若找不到產生器，論文應寫成「由輸出檔案反向驗證」，而不是「程式以隨機方式產生」。

## B. Augmentation 類型再確認

| 類型 | 階段 | 是否正式實驗 | 是否只作用於 train | 是否增加資料數量 | 是否 online |
|---|---|---|---|---|---|
| fixed brightness | ROI balanced feature cache / Phase2 train manifest | 是，但只能由輸出 manifest 反向驗證 | Phase2 train 有，val/test 無 | 是，balanced manifest 中 469 張 | 否，屬離線已產生影像 |
| Gaussian blur | Phase1 distillation、Phase1 patch distillation、Phase2 ROI fine-tuning | 是 | 是 | 否 | 是 |
| Gaussian noise | Phase1 distillation、Phase1 patch distillation、Phase2 ROI fine-tuning | 是 | 是 | 否 | 是 |
| Full-image augmentation | Full-image multilabel patch transfer | 正式設定為無 augmentation | 不適用 | 否 | 否 |

來源包含 `phase1_config.json`、`phase1_patch_config.json`、`shared_phase2_finetune_config.json`、`fairness_audit.json`、`training_config.json` 與對應 training scripts。

## C. 正式模型再確認

| 角色 | 正式 checkpoint | 第二輪結論 |
|---|---|---|
| ROI ImageNet Baseline | `outputs/raddino_convnext_tiny_experiment_seed42/phase2_baseline_imagenet/checkpoints/best.pt` | 正式 ROI baseline |
| ROI RAD-DINO CLS Proposed | `outputs/raddino_convnext_tiny_experiment_seed42/phase2_proposed_distilled/checkpoints/best.pt` | 正式 CLS distilled 模型 |
| ROI RAD-DINO Patch Proposed | `outputs/raddino_convnext_tiny_patch_experiment_seed42/phase2_proposed_patch_distilled/checkpoints/best.pt` | 正式 Patch distilled 模型 |
| Full-image 初始化來源 | `outputs/raddino_convnext_tiny_patch_experiment_seed42/phase2_proposed_patch_distilled/checkpoints/patch_proposed_convnext_tiny_5class.pt` | 由 Patch Proposed 匯出並用於 Full-image 初始化 |
| Full-image training best | `outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/checkpoints/best.pt` | Full-image 訓練最佳 checkpoint |
| Demo 實際載入 | `outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/checkpoints/full_image_multilabel_patch_transfer.pt` | Demo 部署 checkpoint |

不可把其他存在但未部署的 checkpoint 寫成最終 Demo 模型。

## D. 正式指標再確認

- ROI formal metrics 來源：`outputs/raddino_convnext_tiny_three_model_comparison_seed42/tables/overall_metrics_comparison.csv`。
- ROI bootstrap 來源：`outputs/raddino_convnext_tiny_three_model_comparison_seed42/statistics/cluster_bootstrap_results.csv`。
- Full-image validation threshold 來源：`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/validation_selected_thresholds.json`。
- Full-image test metrics 來源：`outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/test_metrics.json`。
- ROI Accuracy 是單標籤 top-1 Accuracy。
- Full-image Exact Subset Accuracy 是多標籤全標籤完全一致比例，不等於 ROI Accuracy。
- Bootstrap 結果不支持主要模型差異達統計顯著。

