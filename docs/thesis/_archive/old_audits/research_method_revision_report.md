# Research Method Revision Report v1

## 本次修訂範圍

- 更新第三章 3.16「實驗環境」。
- 將使用者確認的硬體與核心軟體版本寫入正式實驗環境表。
- 先前誤植的套件版本未寫入論文、實驗環境表、evidence trace 或 revision report。
- 未修改任何模型、checkpoint、threshold、Dataset、正式 metrics 或其他研究數字。

## 新增硬體與軟體環境資訊

| 項目 | 正式採用內容 | 來源 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | 使用者確認 |
| GPU VRAM | 約 8 GB | 使用者確認 |
| CPU | 13th Gen Intel Core i7-13620H | 使用者確認 |
| RAM | 16 GB | 使用者確認 |
| Python | 3.12 | 使用者確認 |
| PyTorch | 2.6.0 | 使用者確認 |
| CUDA | 12.4 | 使用者確認 |

## 未納入的套件資訊

- 本次僅寫入使用者最新正式確認的環境項目。
- 其他套件及版本若無正式證據，不寫入論文主要實驗環境表。
- 先前誤植的套件版本已排除，未寫入 DOCX、trace 或本報告。

## 仍未確認的環境項目

- 作業系統正式訓練版本。
- CUDA driver 版本。
- cuDNN 版本。
- 其他未列入主要環境表之套件版本若需作為正式環境資訊，需另行確認或引用正式輸出紀錄。

## 是否更動其他研究數字

- 未更動第三章以外的研究方法數字。
- 未更動第四章正式實驗結果數字。
- 未更動 ROI、Full-image、threshold、bootstrap 或 Demo 相關正式 metrics。

## DOCX 結構與安全檢查

- DOCX OOXML 可解析：PASS。
- `thesis_evidence_trace_v1.csv` 已更新，rows = 22。
- 疑似 ????、????、????、????、???? 掃描：PASS。
- Windows 私人絕對路徑掃描：PASS。
- `????` 亂碼掃描：PASS。
- LibreOffice/soffice 未作為本次必要修改條件；本次以 OOXML 結構檢查與內容掃描完成驗證。

更新時間：2026-07-23T19:53:29
