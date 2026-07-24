# 02 Code Analysis

建立時間：2026-07-23T04:33:24.534294+00:00

本文件只根據程式碼與正式輸出檔做靜態分析；沒有匯入可能啟動模型的正式模組，沒有訓練，也沒有覆蓋既有成果。正式性依據為 launcher/import、checkpoint/config/metrics/audit 互相指向，而非檔名。

## 重要程式深讀摘要
### `app_full_image_multilabel_ollama_gradio.py`
- 用途：正式 Full-image 多標籤 Gradio Demo；串接模型推論、Ground Truth、Ollama、session 與 PDF/MD 報告。
- Classes：_UnavailableOllama
- Functions：utc_now, timestamp_slug, sha256_file, atomic_write_text, atomic_write_json, atomic_write_csv, resolve_path, dependency_versions, port_available, find_port_pids, select_port, check_output_write_permission, patient_info, prediction_csv_row, pure_model_prediction, ground_truth_payload, generate_report_safe, predicted_label_names, build_followup_guidance, ensure_disclaimer, append_followup_guidance, markdown_to_plain_text, report_font, wrapped_pdf_lines, write_report_pdf, html_text, paragraph_html, patient_display_value, predicted_pairs, report_payload_from_prediction
- 主要 imports：PIL, __future__, argparse, csv, datetime, full_image_multilabel_inference_service, full_image_multilabel_ollama_service, full_image_multilabel_report_prompt, gradio, hashlib, html, importlib.metadata, inspect, json, numpy, os, pandas, pathlib
- CLI/參數線索："--project-root", type=Path, default=PROJECT_DEFAULT; "--model", type=Path, default=DEFAULT_MODEL_RELATIVE; "--thresholds", type=Path, default=DEFAULT_THRESHOLDS_RELATIVE; "--catalog", type=Path, default=DEFAULT_CATALOG_RELATIVE; "--output-dir", type=Path, default=DEFAULT_OUTPUT_RELATIVE; "--device", default="auto"
- 關鍵證據字串：softmax, Ollama, Ground Truth, PDF, ImageNet

### `run_project3_demo.ps1`
- 用途：固定啟動正式 Gradio Demo 的 PowerShell launcher。
- Classes：無頂層 class
- Functions：無頂層 function
- 主要 imports：無
- CLI/參數線索：無 argparse 或未偵測
- 關鍵證據字串：未偵測到指定關鍵字

### `src/full_image_multilabel_inference_service.py`
- 用途：正式 Full-image ConvNeXt-Tiny 多標籤推論服務；驗證 checkpoint/threshold/catalog，輸出 Sigmoid 機率與 GT 比對。
- Classes：FullImageMultilabelInferenceService
- Functions：_resolve_device, _hash_pil, _names, get_inference_service, create_probability_figure
- 主要 imports：PIL, __future__, csv, full_image_multilabel_report_prompt, hashlib, infer_full_image_224_multilabel_single, io, json, math, matplotlib, matplotlib.pyplot, numpy, pathlib, threading, time, torch, typing
- CLI/參數線索：無 argparse 或未偵測
- 關鍵證據字串：torch.sigmoid, softmax, Ground Truth, ImageNet

### `src/full_image_multilabel_ollama_service.py`
- 用途：本機 Ollama 說明服務；只接收單張模型結構化結果，驗證保守語氣與 disclaimer。
- Classes：FullImageOllamaError, FullImageMultilabelOllamaService
- Functions：無頂層 function
- 主要 imports：__future__, full_image_multilabel_report_prompt, hashlib, json, time, typing, urllib.error, urllib.parse, urllib.request
- CLI/參數線索：無 argparse 或未偵測
- 關鍵證據字串：Ollama, Ground Truth

### `src/full_image_multilabel_report_prompt.py`
- 用途：Ollama prompt 單一來源；限制只能描述五類 probability、threshold 與模型結果。
- Classes：無頂層 class
- Functions：_clean, build_report_messages, prompt_schema_sha256
- 主要 imports：__future__, hashlib, json, typing
- CLI/參數線索：無 argparse 或未偵測
- 關鍵證據字串：Ollama

### `src/prepare_full_image_224_multilabel_dataset.py`
- 用途：建立 590 張 full-image 五類多標籤 manifest 與 multilabel split。
- Classes：無頂層 class
- Functions：utc_now, sha256_file, replace_atomic, atomic_write_text, atomic_write_json, atomic_write_csv, read_annotations, index_images, inspect_image, build_master, iterative_split, repair_split_sizes, audit_splits, parse_args, main
- 主要 imports：PIL, __future__, argparse, collections, csv, datetime, hashlib, iterstrat.ml_stratifiers, json, numpy, os, pathlib, sys, typing
- CLI/參數線索："--project-root", type=Path, required=True; "--images-dir", type=Path, required=True; "--annotations", type=Path, required=True; "--output-dir", type=Path, required=True; "--seed", type=int, default=42; "--image-size", type=int, default=224
- 關鍵證據字串：MultilabelStratifiedShuffleSplit, ImageNet

### `src/train_full_image_224_multilabel_patch_transfer.py`
- 用途：以 ROI Patch Proposed ConvNeXt-Tiny 初始化 full-image 五類多標籤模型並 fine-tune。
- Classes：FullImageTransform, FullImageMultilabelDataset, FullImageMultilabelConvNeXt
- Functions：utc_now, sha256_file, replace_atomic, atomic_write_text, atomic_write_json, atomic_write_csv, atomic_save_checkpoint, atomic_save_figure, set_seed, seed_worker, initialize_from_roi_export, read_manifest, dataset_integrity, make_loader, shutdown_loader, build_optimizer, compute_threshold_free, evaluate, train_epoch, select_thresholds, threshold_metrics, environment_info, checkpoint_payload, plot_training, plot_test_results, run_smoke_test, parse_args, resolve_args, main
- 主要 imports：PIL, __future__, argparse, csv, datetime, hashlib, infer_patch_proposed_single_roi, json, math, matplotlib, matplotlib.pyplot, numpy, os, pathlib, platform, random, sklearn, sklearn.metrics
- CLI/參數線索："--project-root", type=Path, required=True; "--initialization-checkpoint", type=Path, required=True; "--train-manifest", type=Path, required=True; "--val-manifest", type=Path, required=True; "--test-manifest", type=Path, required=True; "--output-dir", type=Path, required=True
- 關鍵證據字串：BCEWithLogitsLoss, AdamW, CosineAnnealingLR, torch.sigmoid

### `src/build_full_image_ground_truth_catalog.py`
- 用途：建立 Demo 用 Ground Truth catalog、demo images 與 sidecar JSON。
- Classes：無頂層 class
- Functions：parse_args, utc_now, sha256_file, aggregate_source_sha, write_csv_bom, write_json, format_label_vector, label_code, inspect_sources, distribution_data, select_demo_records, manifest_rows, create_lookup, create_distribution_rows, create_class_index, demo_filename, copy_demo_set, font, create_preview, image_src_for_gallery, create_gallery, create_readme, validate_expected, build_outputs, main
- 主要 imports：PIL, __future__, argparse, collections, csv, datetime, hashlib, html, json, os, pathlib, random, shutil, sys, typing
- CLI/參數線索："--project-root", type=Path, required=True; "--images-dir", type=Path, required=True; "--annotations", type=Path, required=True; "--output-dir", type=Path, required=True; "--seed", type=int, default=42; "--demo-count", type=int, default=30
- 關鍵證據字串：Ollama, Ground Truth

### `src/infer_full_image_224_multilabel_single.py`
- 用途：重要支援程式；需搭配正式輸出判定。
- Classes：無頂層 class
- Functions：parse_args, utc_now, timestamp_slug, sha256_file, atomic_write_text, atomic_write_json, atomic_write_csv, normalize_mapping, resolve_model_path, resolve_threshold_path, load_thresholds, validate_preprocessing, validate_checkpoint, resolve_device, validate_ground_truth, validate_image, preprocess_image, sample_metrics, names, atomic_save_visualization, environment_payload, protected_paths, hash_protected, prediction_payload, csv_row, main
- 主要 imports：PIL, __future__, argparse, csv, datetime, hashlib, json, math, matplotlib, matplotlib.pyplot, numpy, os, pathlib, platform, shutil, sys, textwrap, time
- CLI/參數線索："--project-root", type=Path, required=True; "--model", type=Path; "--thresholds", type=Path; "--image", type=Path, required=True; "--output-dir", type=Path; "--device", default="auto"
- 關鍵證據字串：torch.sigmoid, softmax, Ground Truth

### `src/audit_balanced_roi_and_build_manifest.py`
- 用途：稽核 balanced ROI 224 資料夾，建立 4,725 張 ROI manifest，標記 brightness augmented ROI。
- Classes：無頂層 class
- Functions：sha256_file, read_csv_rows, atomic_write_text, atomic_write_json, atomic_write_csv, get_git_commit, parse_filename, collect_source_metadata, audit_and_build, main
- 主要 imports：PIL, __future__, argparse, collections, csv, datetime, hashlib, json, pathlib, platform, re, subprocess, sys, typing
- CLI/參數線索："--project-root", required=True, type=Path, help="Project root, e.g. C:\\Users\\09688\\thoracic-cxr-project-3"
- 關鍵證據字串：未偵測到指定關鍵字

### `src/create_phase2_grouped_split.py`
- 用途：以 source image/SHA group 建立 ROI phase2 split，防止同源洩漏。
- Classes：UnionFind
- Functions：utc_now, sha256_file, parse_bool, read_csv, atomic_write_csv, atomic_write_text, atomic_write_json, validate_balanced_manifest, scan_original_rois, validate_formal_sources, build_sha_super_groups, choose_subset_by_size, optimize_split, build_manifests, values_crossing_splits, audit_leakage, build_count_tables, perform_analysis, dry_run_payload, write_outputs, run, build_parser, main
- 主要 imports：PIL, __future__, argparse, collections, csv, datetime, hashlib, json, math, numpy, os, pathlib, random, re, sys, typing
- CLI/參數線索："--project-root", default=str(project_root;  "--balanced-manifest", default=str( project_root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42" / "roi_manifest.csv" ;  "--original-roi-dir", default=str(project_root / "data" / "processed" / "bbox_crops_224";  "--output-dir", default=str( project_root / "outputs" / "raddino_convnext_tiny_experiment_seed42" / "phase2_split" ; "--seed", type=int, default=42; "--dry-run", action="store_true"
- 關鍵證據字串：未偵測到指定關鍵字

### `src/cache_raddino_teacher_features.py`
- 用途：使用 frozen microsoft/rad-dino 抽 CLS/pooler features，建立 4,725 x 768 teacher cache。
- Classes：無頂層 class
- Functions：utc_now, sha256_file, read_manifest, atomic_write_text, atomic_write_json, validate_preconditions, environment_info, load_teacher, open_images, forward_batch, smoke_test, verify_saved_features, run, build_parser, main
- 主要 imports：PIL, __future__, argparse, csv, datetime, hashlib, json, os, pathlib, platform, sys, time, torch, transformers, typing
- CLI/參數線索： "--project-root", default=r"C:\Users\09688\thoracic-cxr-project-3", ; "--output-dir", default=None; "--batch-size", type=int, default=8; "--progress-every-batches", type=int, default=10; "--local-files-only", action="store_true"; "--smoke-only", action="store_true"
- 關鍵證據字串：未偵測到指定關鍵字

### `src/cache_raddino_teacher_patch_features.py`
- 用途：使用 frozen RAD-DINO patch tokens 建立 4,725 x 768 x 7 x 7 patch teacher cache。
- Classes：ManifestImageDataset
- Functions：utc_now, sha256_file, read_csv, parse_bool_strict, json_default, atomic_write_text, atomic_write_json, atomic_write_csv, atomic_torch_save, set_seed, patch_pair, processor_audit, environment_info, protected_artifact_paths, hash_protected_artifacts, compare_protected_artifacts, guard_output_directory, validate_manifest, collate_images, load_teacher, tensor_input_stats, forward_patch_batch, batch_metric, shard_path, save_shard, load_resume_shards, compute_cache_statistics, verify_loaded_cache, sample_verification, cleanup_resume_shards
- 主要 imports：PIL, __future__, argparse, collections, csv, dataclasses, datetime, gc, hashlib, json, math, numpy, os, pathlib, platform, random, shutil, sys
- CLI/參數線索："--project-root", type=Path, default=root;  "--manifest", type=Path, default=root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42" / "roi_manifest.csv", ;  "--output-dir", type=Path, default=root / "outputs" / "raddino_convnext_tiny_patch_experiment_seed42" / "phase0_patch_teacher_cache", ; "--model-name", default=EXPECTED_MODEL_NAME; "--model-revision", default=EXPECTED_MODEL_REVISION; "--batch-size", type=int, default=32
- 關鍵證據字串：未偵測到指定關鍵字

### `src/smoke_test_raddino_patch_convnext_maps.py`
- 用途：重要支援程式；需搭配正式輸出判定。
- Classes：無頂層 class
- Functions：utc_now, sha256_file, read_csv, parse_bool, json_default, atomic_write_text, atomic_write_json, atomic_write_csv, set_seed, validate_manifest, sample_records, sampled_csv_rows, open_rgb_images, tensor_summary, map_statistics, patch_dimensions, processor_audit, student_preprocessing_audit, environment_info, install_student_hooks, validate_stage_shapes, smoke_forward, write_text_report, run, build_parser, main
- 主要 imports：PIL, __future__, argparse, collections, csv, dataclasses, datetime, hashlib, json, math, numpy, os, pathlib, platform, random, sys, torch, torch.nn.functional
- CLI/參數線索："--project-root", type=Path, default=root; "--manifest", type=Path, default=root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42" / "roi_manifest.csv"; "--output-dir", type=Path, default=root / "outputs" / "raddino_convnext_tiny_patch_experiment_seed42" / "phase0_patch_shape_smoke_test"; "--model-name", default="microsoft/rad-dino"; "--batch-size", type=int, default=2; "--num-samples", type=int, default=10
- 關鍵證據字串：未偵測到指定關鍵字

### `src/train_convnext_tiny_phase1_distillation.py`
- 用途：Phase 1 CLS feature distillation：ConvNeXt-Tiny student 對齊 RAD-DINO CLS cache。
- Classes：StudentTransform, RoiDataset, ConvNeXtTinyStudent
- Functions：utc_now, sha256_file, read_csv, atomic_write_text, atomic_write_json, atomic_write_csv, parse_bool, set_seed, seed_worker, get_random_states, restore_random_states, get_environment, validate_inputs, make_loader, make_optimizer, make_scaler, extract_teacher_batch, check_gradients, one_training_step, auto_probe_batch_size, stage0_smoke_test, create_augmentation_preview, create_directories, log_message, checkpoint_payload, atomic_save_checkpoint, validate_resume, save_plots, read_metrics, export_distilled_backbone
- 主要 imports：PIL, __future__, argparse, collections, csv, datetime, hashlib, json, math, matplotlib, matplotlib.pyplot, numpy, os, pathlib, platform, random, subprocess, sys
- CLI/參數線索："--project-root", default=str(default_root; "--manifest", default=str(cache_root / "roi_manifest.csv"; "--teacher-cache", default=str(cache_root / "teacher_features.pt";  "--output-dir", default=str( default_root / "outputs" / "raddino_convnext_tiny_experiment_seed42" / "phase1_distillation" ; "--epochs", type=int, default=30; "--batch-size", default="auto"
- 關鍵證據字串：AdamW, CosineAnnealingLR, noise_probability, ImageNet

### `src/train_convnext_tiny_phase1_patch_distillation.py`
- 用途：Phase 1 patch feature distillation：ConvNeXt-Tiny final map 對齊 RAD-DINO 7x7 patch map。
- Classes：StudentTransform, RoiDataset, ConvNeXtTinyPatchStudent
- Functions：utc_now, sha256_file, state_dict_sha256, read_csv, replace_with_retry, atomic_write_text, atomic_write_json, atomic_write_csv, atomic_save_checkpoint, atomic_save_figure, parse_bool, set_seed, seed_worker, rng_state, restore_rng_state, environment_info, configure_runtime, validate_manifest, validate_teacher_cache, make_loader, pretrained_weight_info, extract_teacher, patch_alignment, gradient_status, make_optimizer, make_scaler, training_config, protected_paths, hash_paths, compare_hashes
- 主要 imports：PIL, __future__, argparse, collections, csv, datetime, hashlib, io, json, math, matplotlib, matplotlib.pyplot, numpy, os, pathlib, platform, random, sys
- CLI/參數線索："--project-root", type=Path, default=root;  "--manifest", type=Path, default=root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42" / "roi_manifest.csv", ;  "--teacher-cache", type=Path, default=root / "outputs" / "raddino_convnext_tiny_patch_experiment_seed42" / "phase0_patch_teacher_cache" / "teacher_patch_features_7x7.pt", ;  "--output-dir", type=Path, default=root / "outputs" / "raddino_convnext_tiny_patch_experiment_seed42" / "phase1_patch_distillation", ; "--epochs", type=int, default=100; "--minimum-epochs", type=int, default=60
- 關鍵證據字串：AdamW, CosineAnnealingLR, noise_probability, ImageNet

### `src/train_phase2_convnext_tiny_finetune.py`
- 用途：ROI 五類單標籤 Phase 2 fine-tuning；支援 ImageNet、CLS distilled、Patch distilled 初始化。
- Classes：Phase2Transform, RoiClassificationDataset, ConvNeXtTinyClassifier
- Functions：utc_now, sha256_file, read_csv, parse_bool, replace_with_retry, atomic_write_text, atomic_write_json, atomic_write_csv, set_seed, seed_worker, random_states, restore_random_states, environment_info, imagenet_weights_audit, configure_runtime, resolve_device, make_loader, build_optimizer, build_scaler, validate_manifests, locked_config, validate_locked_runtime, ensure_locked_config, resolve_locked_batch, build_fairness_audit, create_output_directories, output_is_nonempty, log_message, create_augmentation_preview, class_metrics
- 主要 imports：PIL, __future__, argparse, collections, csv, datetime, hashlib, json, math, matplotlib, matplotlib.pyplot, numpy, os, pathlib, platform, random, subprocess, sys
- CLI/參數線索："--project-root", type=Path, default=default_root; "--initialization", choices=("distilled", "patch_distilled", "imagenet"; "--train-manifest", type=Path, default=split_root / "train_roi_manifest.csv"; "--val-manifest", type=Path, default=split_root / "val_roi_manifest.csv"; "--test-manifest", type=Path, default=split_root / "test_roi_manifest.csv"; "--shared-protocol", type=Path, default=split_root / "shared_training_protocol.json"
- 關鍵證據字串：CrossEntropyLoss, AdamW, CosineAnnealingLR, softmax, noise_probability, ImageNet

### `src/compare_baseline_cls_patch.py`
- 用途：三模型正式比較：ImageNet Baseline、RAD-DINO CLS Proposed、RAD-DINO Patch Proposed；含 bootstrap 與 McNemar。
- Classes：無頂層 class
- Functions：utc_now, sha256_file, read_json, read_csv, normalized_path, parse_bool, atomic_write_text, atomic_write_json, csv_value, atomic_write_csv, atomic_save_figure, ensure_clean_destination, model_paths, split_audit, enrich_predictions, pair_predictions, read_confusion_csv, metric_integrity, checkpoint_structure, layernorm_audit, fairness_audit, overall_tables, training_tables, per_class_tables, class_error_table, agreement_table, discordant_rows, matrix_rows, cluster_bootstrap, exact_mcnemar
- 主要 imports：__future__, argparse, collections, compare_proposed_vs_baseline, csv, datetime, hashlib, json, math, matplotlib, matplotlib.pyplot, numpy, os, pathlib, platform, re, shutil, sys
- CLI/參數線索："--project-root", type=Path, required=True; "--baseline-dir", type=Path, required=True; "--cls-dir", type=Path, required=True; "--patch-dir", type=Path, required=True; "--split-dir", type=Path, required=True; "--shared-config", type=Path, required=True
- 關鍵證據字串：CrossEntropyLoss, AdamW, CosineAnnealingLR, softmax, ImageNet

### `src/compare_proposed_vs_baseline.py`
- 用途：較舊 Proposed vs Baseline 比較；輔助但不優先於三模型比較。
- Classes：無頂層 class
- Functions：utc_now, sha256_file, read_json, read_csv, atomic_write_text, atomic_write_json, atomic_write_csv, atomic_save_figure, parse_bool, require_files, prediction_key, validate_predictions, confusion_and_metrics, paired_rows, validate_fairness, overall_comparison, per_class_comparisons, cluster_bootstrap, mcnemar, parse_training_wall_seconds, training_efficiency, error_analysis, plot_overall, plot_per_class, plot_training_curves, row_normalize, plot_confusion, plot_confusion_difference, plot_paired_correctness, plot_bootstrap
- 主要 imports：__future__, argparse, collections, csv, datetime, hashlib, json, math, matplotlib, matplotlib.pyplot, numpy, os, pathlib, platform, re, sys, typing
- CLI/參數線索："--project-root", type=Path, default=root; "--proposed-dir", type=Path, default=experiment / "phase2_proposed_distilled"; "--baseline-dir", type=Path, default=experiment / "phase2_baseline_imagenet"; "--split-dir", type=Path, default=experiment / "phase2_split"; "--shared-config", type=Path, default=experiment / "shared_phase2_finetune_config.json"; "--output-dir", type=Path, default=experiment / "final_comparison"
- 關鍵證據字串：CrossEntropyLoss, AdamW, CosineAnnealingLR, ImageNet

### `src/data/collect_raw_images.py`
- 用途：重要支援程式；需搭配正式輸出判定。
- Classes：無頂層 class
- Functions：copy_annotation_csvs, main
- 主要 imports：__future__, collections, common, pathlib, typing
- CLI/參數線索：無 argparse 或未偵測
- 關鍵證據字串：未偵測到指定關鍵字

### `src/data/audit_image_bbox_pairs.py`
- 用途：重要支援程式；需搭配正式輸出判定。
- Classes：無頂層 class
- Functions：main
- 主要 imports：__future__, collections, common, pathlib, typing
- CLI/參數線索：無 argparse 或未偵測
- 關鍵證據字串：未偵測到指定關鍵字

### `src/data/crop_bbox_rois.py`
- 用途：根據 BBox annotation 裁切 ROI；資料處理而非訓練。
- Classes：無頂層 class
- Functions：read_valid_rows, parse_args, main
- 主要 imports：__future__, argparse, common, csv, pathlib, typing
- CLI/參數線索："--margin-ratio", type=float, default=0.0; "--allow-out-of-bounds-clamp", action="store_true"
- 關鍵證據字串：未偵測到指定關鍵字

### `src/data/create_roi_224_master_dataset.py`
- 用途：建立 ROI 224 master dataset/manifest。
- Classes：無頂層 class
- Functions：parse_args, read_csv_rows, norm_path, bool_true, create_master_rows, error_row, copy_master_images, audit_master, write_class_counts, main
- 主要 imports：PIL, __future__, argparse, collections, common, csv, pathlib, shutil, sys, typing
- CLI/參數線索："--model-input-manifest", type=Path, default=metadata_dir(; "--final-model-csv", type=Path, default=metadata_dir(; "--manual-224-review", type=Path, default=metadata_dir(; "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "bbox_crops_224_master"; "--overwrite", action="store_true"
- 關鍵證據字串：未偵測到指定關鍵字

### `src/data/finalize_roi_224_dataset.py`
- 用途：重要支援程式；需搭配正式輸出判定。
- Classes：無頂層 class
- Functions：read_csv_with_fallback, norm_path, add_error, validate_manual_224, write_manual_reports, create_final_reports, main
- 主要 imports：PIL, __future__, collections, common, csv, datetime, json, pathlib, sys, typing
- CLI/參數線索：無 argparse 或未偵測
- 關鍵證據字串：Ground Truth

### `src/data/prepare_model_inputs_224.py`
- 用途：重要支援程式；需搭配正式輸出判定。
- Classes：無頂層 class
- Functions：parse_args, read_rows, bool_true, output_path_for, letterbox_grayscale, main
- 主要 imports：PIL, __future__, argparse, collections, common, csv, pathlib, sys, typing
- CLI/參數線索："--input-csv", type=Path, default=PROJECT_ROOT / "data" / "metadata" / "final_crops_for_model.csv"; "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "bbox_crops_224"; "--image-size", type=int, default=224; "--padding-value", type=int, default=0; "--overwrite", action="store_true", help="Overwrite existing 224 PNG outputs."
- 關鍵證據字串：未偵測到指定關鍵字

### `src/training/create_roi_grouped_split.py`
- 用途：早期/輔助 ROI grouped split；正式 phase2 使用 src/create_phase2_grouped_split.py。
- Classes：無頂層 class
- Functions：read_csv_rows, stable_hash, source_labels, targets, class_targets, assign_splits, write_split_manifests, audit, write_version, main
- 主要 imports：__future__, collections, common, csv, datetime, hashlib, json, pathlib, sys, typing
- CLI/參數線索：無 argparse 或未偵測
- 關鍵證據字串：未偵測到指定關鍵字

## 程式正式性判定
- `app_full_image_multilabel_ollama_gradio.py` 是正式主程式；`run_project3_demo.ps1` 與 `app_startup_audit.json` 確認其啟動與資源路徑。
- `src/full_image_multilabel_inference_service.py` 的 SHA256 驗證、threshold 驗證、`uses_sigmoid=True`、`uses_roi_crop=False` 是 full-image 推論核心證據。
- ROI 正式比較優先採用 `outputs/raddino_convnext_tiny_three_model_comparison_seed42/`，因其包含 paired predictions、overall/per-class metrics、bootstrap 與 fairness audit。
- `_cleanup_quarantine/`、smoke test、demo sessions、早期 docs 僅列為 legacy/test/supporting，不作正式結論的主要來源。
