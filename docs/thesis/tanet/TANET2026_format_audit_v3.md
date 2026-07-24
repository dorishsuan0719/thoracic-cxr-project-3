# TANET2026 Format Audit v3

## Scope

- Source: `docs/thesis/tanet/胸腔X光影像分類與知識蒸餾系統_TANET2026短篇_v2.docx`.
- Output: `docs/thesis/tanet/胸腔X光影像分類與知識蒸餾系統_TANET2026短篇_v3.docx`.
- Dataset, checkpoints, thresholds, metrics, research numbers, citation mapping, source code, and evidence package were not modified.
- Template status: `docs/thesis/templates/TANET_論文格式.docx` was not found in project-3, so v3 follows the requested TANET numeric layout settings and preserves the v2 citation mapping.

## Applied Layout Fixes

- Moved the continuous section break from after English Keywords to immediately after E-mail.
- Section 1 now contains only the Chinese title, English title, author placeholder, affiliation placeholder, and E-mail placeholder.
- Section 2 is two-column and contains the Chinese abstract, keywords, English Abstract, Keywords, body text, tables, and references.
- Removed ordinary page breaks; the document uses no manual page breaks.
- Added column breaks only where needed: before Table 1, Table 4, Table 6, and reference [6].
- Table captions are above tables and use keep-with-next behavior.
- Table rows use non-splitting row settings.
- Table 3 uses `Act.`, `Cross-Entropy`, and `BCE with Logits` to avoid cramped text.
- Table 5 uses `Acc.`, `M-F1`, `W-F1`, and `AUC` headers; metric values and interpretation were not changed.
- Reference formatting was normalized without changing the [1]-[9] citation mapping.

## Structural Checks

- Actual page count from Microsoft Word COM: 4.
- Section count: 2.
- Section columns: Section 1 = 1 column; Section 2 = 2 columns.
- Continuous section break location: after `E-mail: email@example.edu.tw`; the next visible paragraph is `摘要`.
- Manual page break count: 0.
- Column break count: 4.
- Table count: 7.
- Embedded image count: 0.
- Reference count: 9.
- Chinese font in OOXML: 標楷體.
- English/number font in OOXML: Times New Roman.

## Render Inspection

- Render method: Microsoft Word COM page rendering to page images, then visual inspection.
- Page images inspected: 4 pages.
- Table 1: PASS, complete in one column on page 2 left column.
- Table 6: PASS, complete in one column on page 3 right column.
- Table 4: PASS, complete in one column on page 3 left column.
- Table 5: PASS, no numeric value split after final column-width adjustment.
- Final page balance: PASS, references [1]-[5] appear in the left column and [6]-[9] appear in the right column.
- Split table / clipping / overlap check: PASS by visual inspection of rendered pages.

## Content Protection

- Protected research numbers were preserved, including Macro-F1 0.7865, Micro-F1 0.7859, 4,725, 945, 469, Train/Val/Test counts, feature shapes, Phase 1 loss/cosine values, ROI metrics, Full-image metrics, validation thresholds, bootstrap conclusion, and Ground Truth/Ollama role boundary.
- No Dataset, checkpoint, threshold, metric, source code, or evidence-package file was modified.

## Safety Checks

- Private absolute path scan: PASS.
- Mojibake scan: PASS.
- Placeholder scan: PASS.
- Sensitive-pattern scan: PASS.
- Real patient identifier scan: no new patient identifiers were added.

## Remaining Author Fields

- Author name remains a placeholder.
- Affiliation remains a placeholder.
- E-mail remains a placeholder.
