# Thoracic CXR Project 3

Independent five-class BBox ROI crop dataset project.

This project treats `C:\Users\09688\thoracic-cxr-project-3` and `C:\Users\09688\thoracic-cxr-project-3` as read-only sources. It copies annotation CSVs and raw full images into this project, then audits BBox alignment before producing ROI crops.

Run order:

```powershell
python src\data\collect_raw_images.py
python src\data\audit_image_bbox_pairs.py
python src\data\visualize_bbox_alignment.py
python src\data\crop_bbox_rois.py --margin-ratio 0
python src\data\audit_cropped_dataset.py
```

Formal crop is stopped if any audited valid BBox requires image-boundary clamp, unless `--allow-out-of-bounds-clamp` is explicitly provided.

## Run Summary

Generated on 2026-07-15.

Read-only source projects:

- `C:\Users\09688\thoracic-cxr-project-3`
- `C:\Users\09688\thoracic-cxr-project-3`

Annotation CSVs used:

- `C:\Users\09688\thoracic-cxr-project-3\data\splits\five_class_train_annotations.csv`
- `C:\Users\09688\thoracic-cxr-project-3\data\splits\five_class_val_annotations.csv`
- `C:\Users\09688\thoracic-cxr-project-3\data\splits\five_class_test_annotations.csv`

Image search order mirrors the YOLO project's `src\data\visualize_bbox_sanity_check.py`:

1. `C:\Users\09688\thoracic-cxr-project-3\data\raw\dicom_files` - exists, 3320 indexed files
2. `C:\Users\09688\thoracic-cxr-project-3\data\raw\train` - missing, 0 indexed files
3. `C:\Users\09688\thoracic-cxr-project-3\vindr_project\dicom_files` - missing, 0 indexed files
4. `C:\Users\09688\thoracic-cxr-project-3\data\raw` - exists, 0 indexed files

Collected raw images:

- Unique source images: 1540
- Copied/converted to `data\raw\images`: 1540
- Missing images: 0

Audit totals:

- Annotation rows: 2672
- Valid BBox rows after exact duplicate removal: 2493
- Missing images: 0
- Invalid BBoxes: 0
- Exact duplicate annotations removed: 179
- BBox rows needing clamp: 0

Generated outputs:

- BBox overlays: 150 PNG files in `data\processed\bbox_overlay`
- ROI crops: 2493 PNG files in `data\processed\bbox_crops`
- Crop manifest: `data\metadata\crop_manifest.csv`
- Full project tree: `outputs\reports\project_tree.txt`
- Final crop audit: `outputs\reports\final_crop_dataset_audit.txt`

Leakage check:

- train vs val: 0
- train vs test: 0
- val vs test: 0
