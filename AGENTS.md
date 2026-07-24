# 胸腔 X 光專題論文工作規則

## 專案範圍

本 Repository 為正式專案：

C:\Users\09688\thoracic-cxr-project-3

不可引用已刪除或舊版專案：

- thoracic-cxr-project
- thoracic-cxr-yolo-project
- thoracic-cxr-project_OLD_DISABLED

## 安全限制

執行論文整理工作時：

1. 不可修改、刪除或重新訓練模型。
2. 不可修改 checkpoint、threshold、Ground Truth 或正式 metrics。
3. 不可修改正式 Demo 與推論程式。
4. 不可修改資料集標籤、split、image_id 或 SHA256。
5. 不可刪除任何圖片、CSV、JSON 或輸出結果。
6. 所有新文件只能放在 docs/thesis/。
7. 只允許建立 Markdown、CSV、JSON、HTML、Python 輔助程式及 DOCX。
8. 執行前先建立分析報告，不可直接產生最終論文。

## 正式研究方向

研究包含：

1. 依 BBox 製作五類 ROI。
2. 建立平衡後 4,725 張 ROI，每類 945 張。
3. 使用 microsoft/rad-dino 建立 CLS 與 Patch Feature Cache。
4. 將 RAD-DINO 特徵蒸餾至 ConvNeXt-Tiny。
5. 比較 ImageNet Baseline、CLS Distilled、Patch Distilled。
6. 將 ROI Patch Proposed Backbone 轉移至 Full-image 模型。
7. 使用 590 張完整胸腔 X 光進行五類多標籤 Fine-tuning。
8. 整合 Gradio、Ground Truth、Ollama 與 PDF 報告。

## 正式五類

0. Aortic enlargement
1. Cardiomegaly
2. Pleural thickening
3. Pulmonary fibrosis
4. Pleural effusion

沒有 No Finding 類別。

## 不得混淆

1. ROI 階段是五類單標籤分類。
2. Full-image 階段是五類多標籤分類。
3. ROI 模型使用 Softmax。
4. Full-image 模型使用 Sigmoid。
5. 最終 Demo 使用 Full-image ConvNeXt-Tiny。
6. 最終 Demo 不使用 YOLO、BBox 或 ROI Crop。
7. RAD-DINO 是蒸餾 Teacher，不是最終 Demo 推論模型。
8. Ollama 不負責影像分類。
9. Ground Truth 不輸入模型，也不傳給 Ollama。
10. Exact Subset Accuracy 不等於一般單標籤 Accuracy。
11. 不可宣稱 Patch Proposed 全面勝過 Baseline。
12. Bootstrap 95% CI 若跨越 0，不可宣稱具有統計顯著差異。

## 資料可信度順序

數字與研究事實優先順序：

1. 正式 evaluation CSV／JSON。
2. 正式 metrics、audit 與 summary 文件。
3. 正式訓練與評估程式。
4. README 與架構文件。
5. 舊版筆記與註解只能當參考。

若不同檔案有衝突：

- 不可自行選一個數字。
- 必須列出衝突來源。
- 標記為「待使用者確認」。

## 論文規則

1. 不可捏造資料數量、指標、Epoch、Threshold 或訓練設定。
2. 不可捏造作者、論文、期刊、年份、DOI 或網址。
3. 每一個主要數字都需記錄來源檔案。
4. 無法確認的內容使用「〔待確認〕」。
5. 使用正式繁體中文學術語氣。
6. 不可把程式註解當成已驗證的實驗結果。
