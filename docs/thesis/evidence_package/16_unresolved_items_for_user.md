# 16 Unresolved Items For User

本文件只列出第二輪查證後仍無法由目前專案證據完全確認、需要使用者決定或補充資料的項目。論文初稿可先採用保守寫法，不需等待全部項目解決。

## 1. Brightness augmentation 產生器與抽樣策略

### 問題

目前可從 `roi_manifest.csv` 與檔名反向驗證 469 張 brightness augmentation 影像，以及 factor 值介於 0.95 至 1.05。但尚未找到實際產生這些影像的 generator 程式或執行紀錄。

### 可選答案

| 選項 | 內容 | 對論文影響 |
|---|---|---|
| A | 找到 generator 程式或執行紀錄 | 可明確描述產生流程、抽樣策略與 random seed |
| B | 找不到 generator，只採用 manifest 反向驗證 | 只能描述已輸出影像的 factor 與數量，不可描述程式公式 |
| C | 不在論文主文強調 brightness，只列入附錄稽核 | 降低方法章節的疑義 |

### 建議保守寫法

「balanced ROI manifest 中含 469 張可由檔名辨識之 brightness augmentation 影像，其 factor 值介於 0.95 至 1.05；本研究僅能由輸出檔案反向驗證該事實，未將其產生抽樣公式作為已驗證事實。」

## 2. BBox 與 ROI crop 人工審查逐筆結果

### 問題

目前找到 summary 層級的人工審查報告，例如 BBox overlay pass/review 統計與 crop review 統計，但未找到逐筆人工審查 CSV。若論文要描述逐筆排除或逐筆修正原因，需要更細證據。

### 可選答案

| 選項 | 內容 | 對論文影響 |
|---|---|---|
| A | 使用者提供逐筆人工審查 CSV | 可寫更完整的品質控管流程與案例統計 |
| B | 僅使用 summary 報告 | 可寫高層級 QC 統計，不可寫逐筆決策 |
| C | 將人工審查細節移至限制與附錄 | 主文避免過度展開 |

### 建議保守寫法

「本研究曾進行 BBox overlay、crop 與 224 resize 抽樣人工檢查；目前可由 summary 報告確認整體通過與需檢視比例，但逐筆審查紀錄未納入本次證據包，因此論文僅回報 summary 層級結果。」

## 3. 外部測試集或跨院資料泛化

### 問題

目前 evidence package 可確認 train/validation/test split 與正式 test metrics，但未看到獨立外部資料集評估。

### 可選答案

| 選項 | 內容 | 對論文影響 |
|---|---|---|
| A | 補充外部測試資料與正式 metrics | 可討論外部泛化能力 |
| B | 不補外部測試 | 只能說明在既有 split 的 test set 上評估 |
| C | 將外部泛化列為未來工作 | 符合目前證據邊界 |

### 建議保守寫法

「本研究之模型表現係於既有資料切分之 test set 評估；尚未以外部醫院或跨資料集影像驗證泛化能力，後續可加入外部測試集進行確認。」

## 4. Full-image multilabel bootstrap 信賴區間

### 問題

ROI 三模型比較已有 cluster bootstrap 統計，但 Full-image multilabel test metrics 目前未看到 bootstrap 95% CI。若論文要宣稱 Full-image 模型或 Demo 模型在統計上顯著優於其他方法，需要額外證據。

### 可選答案

| 選項 | 內容 | 對論文影響 |
|---|---|---|
| A | 補做 Full-image bootstrap CI | 可在論文中報告信賴區間與不確定性 |
| B | 不補 bootstrap | 只能報告 point estimate，不做顯著性主張 |
| C | 只將 CI 作為未來統計補強 | 主文維持保守 |

### 建議保守寫法

「Full-image multilabel 任務目前回報 test set point estimates；本研究不主張其與其他模型之差異達統計顯著，後續可透過 bootstrap 信賴區間補充不確定性分析。」

## 初稿條件判斷

即使上述四項尚未完全解決，論文已具備撰寫保守初稿的條件。初稿中需避免以下主張：

1. 不宣稱 brightness augmentation 的 generator 為 `random.uniform(0.95, 1.05)`。
2. 不宣稱具備逐筆人工審查完整紀錄。
3. 不宣稱模型已通過外部測試集驗證。
4. 不宣稱 Full-image multilabel 結果具有統計顯著優勢。

