# 08 Augmentation Audit

建立時間：2026-07-23T04:33:24.534294+00:00

## 搜尋範圍與關鍵字

靜態搜尋範圍包含 `src/`、正式 app、正式 config/audit JSON 與主要 Markdown；排除 raw annotations 中與 augmentation 無關的座標/ID 數字碰撞。關鍵字包含 `469`, `brightness`, `__aug_brightness`, `ImageEnhance`, `ColorJitter`, `RandomAffine`, `RandomRotation`, `GaussianBlur`, `noise`, `contrast`, `augment`, `is_brightness_augmented`。

## 469 brightness augmentation

- 結論：469 張 brightness augmented ROI 已確認存在於正式 balanced ROI manifest。
- 證據檔：`outputs/raddino_feature_cache/balanced_945_seed42/roi_manifest.csv`。
- 每類 augmented rows：{'Aortic enlargement': 173, 'Cardiomegaly': 162, 'Pleural effusion': 134}。
- 檔名 factor 分布：{0.95: 22, 0.96: 54, 0.97: 73, 0.98: 57, 0.99: 26, 1.01: 21, 1.02: 53, 1.03: 63, 1.04: 75, 1.05: 25}。
- factor 解讀：`f095` 到 `f105` 對應 0.95 到 1.05，包含較暗與較亮。這是從正式 manifest/filename 可驗證的 factor，不是重新執行產生器推測。
- 限制：本次掃描未找到產生這 469 張影像的原始 generator 程式；產生演算法位置標記為 `〔待確認〕`。
- split：`src/create_phase2_grouped_split.py` 驗證 val/test brightness count 必須為 0；正式 split 為 train 357、val 0、test 0。

## Online augmentation: ROI Phase 1/2

- `src/train_convnext_tiny_phase1_distillation.py` 與 `src/train_convnext_tiny_phase1_patch_distillation.py` 使用 Gaussian blur p=0.20 與 Gaussian noise p=0.30；noise std 0.005-0.015。
- `brightness_transform=false`、`contrast_transform=false`。
- `src/train_phase2_convnext_tiny_finetune.py` 使用相同風格 blur/noise；`shared_phase2_finetune_config.json` 記錄 train-only、brightness=false、contrast=false、color_jitter=false。
- `fairness_audit.json` 確認三個 ROI Phase 2 模型 augmentation locked、train-only、augmentation preview SHA256 identical。

## ColorJitter / BBox-aware augmentation / Full-image augmentation

- 未發現正式 `ColorJitter`；正式 config 顯示 `color_jitter=false`。
- `src/data/crop_bbox_rois.py` 是 deterministic BBox crop；未發現 rotation/translation/scale/flip 並同步更新 BBox 的正式 augmentation pipeline。
- Full-image multilabel training config 與 preprocessing spec 均記錄 `augmentation=false`；Full-image manifest integrity 禁止 `bbox_crops`、`roi_balanced`、`augmentation` path。
