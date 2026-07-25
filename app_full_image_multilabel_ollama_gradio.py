#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full-image five-class multilabel Gradio demo with local Ollama explanation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import re
import threading
import textwrap
import time
import traceback
import uuid
from datetime import datetime, timezone
from urllib.parse import quote
from pathlib import Path
from typing import Any, Iterator


PROJECT_DEFAULT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DEFAULT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from full_image_multilabel_inference_service import (
    DEFAULT_CATALOG_RELATIVE,
    DEFAULT_MODEL_RELATIVE,
    DEFAULT_THRESHOLDS_RELATIVE,
    EXPECTED_FORMAL_MODEL_SHA256,
    EXPECTED_THRESHOLD_SHA256,
    FullImageMultilabelInferenceService,
    create_probability_figure,
    get_inference_service,
)
from full_image_multilabel_ollama_service import (
    FullImageMultilabelOllamaService,
    FullImageOllamaError,
)
from full_image_multilabel_report_prompt import (
    CLASS_MAPPING_EN,
    CLASS_MAPPING_ZH,
    DISCLAIMER,
    build_report_messages,
    prompt_schema_sha256,
)


DEFAULT_OUTPUT_RELATIVE = Path("outputs/full_image_multilabel_gradio_demo")
DEFAULT_VAL_MANIFEST = Path(
    "outputs/full_image_224_multilabel_seed42/phase0_dataset/val_manifest.csv"
)
DEFAULT_TEST_MANIFEST = Path(
    "outputs/full_image_224_multilabel_seed42/phase0_dataset/test_manifest.csv"
)
ANALYSIS_OUTPUT_COUNT = 14
REGENERATE_OUTPUT_COUNT = 6
PROBABILITY_TABLE_COLUMNS = ["類別", "英文名稱", "中文名稱", "機率", "Threshold", "判定"]
_UI_ERROR_LOCK = threading.Lock()
OLLAMA_FALLBACK = "分類已完成，但 Ollama 輔助說明暫時無法產生。"
UI_LAYOUT_VERSION = "2026-07-25-stable-ui-clinical-reference-v7-clean-report"
GROUND_TRUTH_UNAVAILABLE = (
    "此影像不在目前 Ground Truth Catalog 中，因此僅顯示模型預測，不計算正確性。"
)
APP_CSS = """
:root {
  --page: #ffffff;
  --surface: #ffffff;
  --header: #faf7ff;
  --header-strong: #f3edff;
  --lavender: #d9cbff;
  --lavender-line: #e6ddfb;
  --ink: #151b2b;
  --muted: #697386;
  --line: #dce2ea;
  --teal: #16b7a4;
  --teal-dark: #0d9488;
  --blue: #4796e8;
  --yellow: #ffc928;
  --red: #e63e4f;
  --green: #16a34a;
  --soft-green: #ecfdf3;
  --soft-yellow: #fff8dc;
  --soft-red: #fff1f2;
  --slate: #64748b;
  --slate-dark: #52606d;
  --soft-teal: #e6f8f5;
  --purple: #9a67f5;
  --purple-dark: #7c4fe0;
  --soft-purple: #f5efff;
}

* { box-sizing: border-box !important; }

html, body, .gradio-container {
  background: var(--page) !important;
  color: var(--ink) !important;
  font-family: "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", Arial, sans-serif !important;
}

.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap {
  background: transparent !important;
  max-width: none !important;
}

.app-shell {
  width: min(1500px, calc(100% - 36px));
  margin: 0 auto;
  padding: 18px 0 34px;
}

.topbar-host,
.topbar-host > div,
.card-header-host,
.card-header-host > div,
.report-html-host,
.report-html-host > div,
.system-html-host,
.system-html-host > div {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  margin: 0 !important;
  padding: 0 !important;
}

.app-topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  border-bottom: 2px solid #eadffc;
  padding: 4px 0 16px;
  margin-bottom: 12px;
}

.app-title-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.app-topbar h1 {
  margin: 0 !important;
  color: var(--ink) !important;
  font-size: clamp(1.65rem, 2.6vw, 2.25rem);
  line-height: 1.2;
  font-weight: 900;
  background: transparent !important;
}

.app-topbar p {
  margin: 6px 0 0 !important;
  color: var(--muted) !important;
  font-size: .95rem;
  background: transparent !important;
}

.research-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  background: var(--purple) !important;
  color: #fff !important;
  padding: 4px 10px;
  font-weight: 800;
  font-size: .78rem;
  white-space: nowrap;
}

.top-health {
  min-width: 178px;
  border: 1px solid #d9efe9;
  border-radius: 10px;
  background: #f8fffc !important;
  padding: 7px 10px;
  text-align: left;
  font-size: .8rem;
  line-height: 1.55;
  display: flex;
  align-items: center;
}

.health-ok { color: #08785f !important; font-weight: 800; }
.health-dot { color: #12b886 !important; margin-right: 5px; }
.header-icon { vertical-align: -3px; margin-right: 6px; color: var(--purple-dark); }

.ui-card {
  background: var(--surface) !important;
  border: 1px solid var(--lavender-line) !important;
  border-radius: 11px !important;
  box-shadow: 0 2px 8px rgba(45, 55, 72, .045) !important;
  overflow: hidden !important;
  padding: 0 !important;
  margin: 0 0 12px !important;
  width: 100% !important;
}

.ui-card-header {
  width: 100%;
  min-height: 35px;
  background: linear-gradient(90deg, #fbf8ff 0%, #f5efff 100%) !important;
  border: 0 !important;
  border-bottom: 1px solid #eee7fb !important;
  padding: 8px 13px !important;
  margin: 0 !important;
}

.ui-card-header,
.ui-card-header * {
  box-shadow: none !important;
}

.ui-card-header h3 {
  margin: 0 !important;
  color: var(--ink) !important;
  font-size: 1rem;
  line-height: 1.35;
  font-weight: 900;
  background: transparent !important;
}

.ui-card-header p {
  margin: 3px 0 0 !important;
  color: var(--muted) !important;
  font-size: .79rem;
  line-height: 1.4;
  background: transparent !important;
}

.ui-card-body {
  width: 100%;
  background: #fff !important;
  border: 0 !important;
  box-shadow: none !important;
  margin: 0 !important;
  padding: 12px 14px !important;
}

.ui-card-body > .gr-row,
.ui-card-body > .gr-column,
.ui-card-body .gr-row,
.ui-card-body .gr-column {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  min-width: 0 !important;
}

.patient-card .ui-card-body { padding: 9px 13px 12px !important; }
.patient-fields {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important;
  gap: 12px !important;
}
.patient-note-field { grid-column: span 2 !important; min-width: 0 !important; }
@media (max-width: 640px) {
  .patient-note-field { grid-column: span 1 !important; }
}

.main-grid {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: flex-start !important;
  gap: 18px !important;
}

.left-pane,
.right-pane {
  flex: 1 1 0 !important;
  width: 0 !important;
  min-width: 0 !important;
  max-width: none !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: #fff !important;
  color: var(--ink) !important;
  border: 1px solid #cfd7e2 !important;
  border-radius: 7px !important;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .secondary-text {
  color: #445065 !important;
  background: transparent !important;
}

.image-card {
  width: 100% !important;
  margin: 0 0 12px !important;
  padding: 0 !important;
  background: #fff !important;
  border: 1px solid var(--lavender-line) !important;
  border-radius: 11px !important;
  box-shadow: 0 2px 8px rgba(45, 55, 72, .045) !important;
  overflow: hidden !important;
}

.image-card .ui-card-header {
  min-height: auto !important;
  margin: 0 !important;
  padding: 4px 0 10px !important;
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}

.image-card .ui-card-header h3 {
  margin: 0 !important;
  line-height: 1.35 !important;
}

.image-card .card-header-host,
.image-card .card-header-host > div {
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}

.image-host {
  padding: 8px 14px 0 !important;
}

.image-host,
.image-host > div {
  background: #fff !important;
  border: 0 !important;
  box-shadow: none !important;
}

.image-host .image-container,
.image-host .image-frame,
.image-host img {
  max-width: 100% !important;
}

#cxr-image-input > .label-wrap,
#cxr-image-input .label-wrap,
#cxr-image-input > label {
  display: none !important;
}

/* Stable Accordion styling: target our own elem_classes instead of Gradio's
   internal .accordion / .label-wrap class names, which can change by version. */
.app-accordion {
  background: #fff !important;
  border: 1px solid var(--lavender-line) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  overflow: hidden !important;
}
.app-accordion > button,
.app-accordion > summary,
.app-accordion > .label-wrap,
.app-accordion > [role="button"] {
  width: 100% !important;
  min-height: 38px !important;
  background: linear-gradient(90deg, #fbf8ff 0%, #f5efff 100%) !important;
  border: 0 !important;
  border-radius: 0 !important;
  padding: 8px 12px !important;
  color: var(--ink) !important;
  font-weight: 800 !important;
  box-shadow: none !important;
}
.app-accordion > button:hover,
.app-accordion > summary:hover,
.app-accordion > .label-wrap:hover,
.app-accordion > [role="button"]:hover {
  background: #f3edff !important;
}
.gt-accordion.ui-card { border: 1px solid var(--lavender-line) !important; }
.gt-accordion > button,
.gt-accordion > summary,
.gt-accordion > .label-wrap,
.gt-accordion > [role="button"] {
  min-height: 35px !important;
  padding: 8px 13px !important;
  font-size: 1rem !important;
  font-weight: 900 !important;
}
.gt-accordion .ui-card-body { border-top: 1px solid #eee7fb !important; }
.gt-hint { color: var(--muted) !important; font-size: .78rem; margin: 2px 0 0 !important; }

.action-row {
  gap: 12px !important;
  padding: 10px 14px 13px !important;
  margin: 0 !important;
  border-top: 1px solid #edf0f4 !important;
  background: #fff !important;
  flex-wrap: wrap !important;
}
.action-row > * { flex: 1 1 130px !important; }

.gradio-container button {
  min-height: 40px !important;
  border-radius: 7px !important;
  font-weight: 900 !important;
}
.primary-action, .primary-action button, button.primary-action, .gradio-container button.primary {
  background: var(--teal) !important; border-color: var(--teal) !important; color: #fff !important;
}
.primary-action:hover, .primary-action button:hover { background: var(--teal-dark) !important; }
.secondary-action, .secondary-action button { background: var(--blue) !important; border-color: var(--blue) !important; color: #fff !important; }
.print-action, .print-action button, .sheet-print-button, .sheet-print-button button { background: var(--slate) !important; border-color: var(--slate-dark) !important; color: #fff !important; }
.print-action:hover, .print-action button:hover, .sheet-print-button:hover, .sheet-print-button button:hover { background: var(--slate-dark) !important; }
.danger-action, .danger-action button { background: #fff !important; border-color: #cfd7e2 !important; color: #4a5568 !important; }
.danger-action:hover, .danger-action button:hover { background: #f5f6f8 !important; border-color: var(--red) !important; color: var(--red) !important; }

.transparent-output,
.transparent-output > div,
.transparent-output .prose,
.transparent-output .markdown-text,
.transparent-output .output-markdown,
.transparent-output p,
.transparent-output span,
.transparent-output strong,
.transparent-output ul,
.transparent-output ol,
.transparent-output li,
.transparent-output h1,
.transparent-output h2,
.transparent-output h3,
.transparent-output h4 {
  background: transparent !important;
  background-color: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.transparent-output { color: var(--ink) !important; line-height: 1.62 !important; padding: 0 !important; }

.analysis-status .prose { margin: 0 !important; }
.analysis-status .status-complete { color: var(--green) !important; font-size: 1.03rem; font-weight: 900; margin-bottom: 8px; }
.analysis-status .status-grid { display: grid; grid-template-columns: 136px minmax(0,1fr); gap: 5px 8px; font-size: .86rem; }
.analysis-status .status-label { font-weight: 800; color: #303b4d !important; }

.summary-card .ui-card-body { min-height: 136px; }
.prediction-summary ul { margin: 6px 0 8px 1.2rem !important; padding: 0 !important; }
.prediction-summary li::marker { color: #1dbd7f; }
.prediction-summary .vector-note { border-top: 1px solid #e6eaf0; margin-top: 10px; padding-top: 9px; font-size: .82rem; color: var(--muted) !important; }

/* Stable probability table: this HTML and these classes are generated by us,
   so row coloring does not depend on Gradio Dataframe DOM internals or JS. */
.probability-card-body {
  padding: 3px 14px 12px !important;
}

#probability-table-host,
#probability-table-host > div,
#probability-table-host .prose,
#probability-table-host .html-container,
.probability-table-host,
.probability-table-host > div {
  width: 100% !important;
  max-width: 100% !important;
  min-height: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

#probability-table-host .prose,
#probability-table-host .html-container {
  margin-block: 0 !important;
}

/* 外框使用覆蓋層繪製，避免被表格內容蓋住四個圓角。 */
.probability-table-frame {
  position: relative !important;
  isolation: isolate !important;
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  background: #fff !important;
  border: 0 !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  box-shadow: none !important;
}

.probability-table-frame::after {
  content: "" !important;
  position: absolute !important;
  inset: 0 !important;
  z-index: 10 !important;
  pointer-events: none !important;
  border: 1px solid #1f2937 !important;
  border-radius: 8px !important;
  box-sizing: border-box !important;
}

/* 只有內容負責水平捲動，固定外框維持在最上層。 */
.probability-table-wrap {
  position: relative !important;
  z-index: 1 !important;
  width: 100% !important;
  max-width: 100% !important;
  overflow-x: auto !important;
  overflow-y: hidden !important;
  background: #fff !important;
  border: 0 !important;
  border-radius: 8px !important;
  margin: 0 !important;
  padding: 0 !important;
}

.probability-table {
  width: 100% !important;
  border-collapse: collapse !important;
  border-spacing: 0 !important;
  table-layout: auto !important;
  background: #fff !important;
  font-size: .83rem !important;
  border: 0 !important;
  border-radius: 0 !important;
}

.probability-table th,
.probability-table td {
  padding: 8px 9px !important;
  border: 0 !important;
  border-right: 1px solid #d8dee8 !important;
  border-bottom: 1px solid #d8dee8 !important;
  text-align: left !important;
  white-space: nowrap !important;
}

.probability-table th:last-child,
.probability-table td:last-child {
  border-right: 0 !important;
}

.probability-table tbody tr:last-child td {
  border-bottom: 0 !important;
}
.probability-table th {
  background: #f5f7fa !important;
  color: #334155 !important;
  font-weight: 900 !important;
}
.probability-table td { background: #fff !important; color: var(--ink) !important; }
.probability-table tr.prob-positive td { background: var(--soft-red) !important; }
.probability-table tr.prob-positive td:last-child { color: var(--red) !important; font-weight: 900 !important; }
.probability-table tr.prob-negative td:last-child { color: #8a94a6 !important; font-weight: 700 !important; }
.probability-table-empty {
  border: 1px dashed #d7dde6 !important;
  border-radius: 7px !important;
  padding: 16px !important;
  color: var(--muted) !important;
  text-align: center !important;
  background: #fafbfc !important;
}

.gt-compact .gt-match { color: var(--green) !important; font-weight: 900; }
.gt-compact code { background: #f4f6f9 !important; padding: 1px 4px !important; }

.report-card { box-shadow: 0 4px 18px rgba(32, 41, 58, .07) !important; }
.report-toolbar { display: flex !important; align-items: center !important; gap: 12px !important; }
.report-toolbar .header-copy { flex: 1 1 auto; min-width: 0; }
.report-toolbar .sheet-print-button { flex: 0 0 auto; min-width: 150px; }
.report-body { padding: 12px 14px 14px !important; background: #f8f9fb !important; }

#diagnosis-sheet {
  width: 100%;
  background: #fff !important;
  border: 1px solid #d9e0e8 !important;
  border-radius: 7px !important;
  box-shadow: 0 2px 8px rgba(30, 41, 59, .05) !important;
  color: var(--ink) !important;
  margin: 0 !important;
  padding: 18px 20px 20px !important;
}

#diagnosis-sheet,
#diagnosis-sheet div,
#diagnosis-sheet p,
#diagnosis-sheet span,
#diagnosis-sheet strong,
#diagnosis-sheet ul,
#diagnosis-sheet ol,
#diagnosis-sheet li,
#diagnosis-sheet h1,
#diagnosis-sheet h2,
#diagnosis-sheet h3,
#diagnosis-sheet h4 { box-shadow: none !important; }

#diagnosis-sheet .sheet-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  border-bottom: 2px solid var(--teal) !important;
  margin-bottom: 14px;
  padding-bottom: 10px;
}
#diagnosis-sheet h2 { margin: 0 0 3px !important; font-size: 1.35rem; line-height: 1.25; color: var(--ink) !important; }
#diagnosis-sheet .sheet-subtitle, #diagnosis-sheet .sheet-meta { color: var(--muted) !important; font-size: .79rem; line-height: 1.45; }
#diagnosis-sheet .research-stamp { border: 2px solid var(--purple) !important; border-radius: 8px !important; background: var(--soft-purple) !important; color: var(--purple-dark) !important; padding: 7px 10px; font-weight: 900; white-space: nowrap; }

#diagnosis-sheet .report-top-grid { display: grid; grid-template-columns: 1.15fr .85fr; gap: 16px; align-items: start; margin-bottom: 12px; }
#diagnosis-sheet .basic-info-box { border-right: 1px solid #e3e7ed !important; padding-right: 15px; }
#diagnosis-sheet .clinical-box { background: var(--soft-teal) !important; border-radius: 7px !important; padding: 10px 12px !important; }
#diagnosis-sheet .clinical-box h3 { margin-top: 0 !important; }
#diagnosis-sheet .clinical-diagnosis-list { margin: 5px 0 0 1.1rem !important; }
#diagnosis-sheet .clinical-diagnosis-list li::marker { color: #d8a700; }

#diagnosis-sheet .sheet-section { border-bottom: 1px solid #e4e8ee !important; margin: 0 0 12px; padding: 0 0 11px; break-inside: avoid; page-break-inside: avoid; }
#diagnosis-sheet .sheet-section:last-of-type { border-bottom: 0 !important; }
#diagnosis-sheet h3 { color: #194f8e !important; font-size: .98rem; line-height: 1.35; margin: 0 0 6px !important; padding: 0 !important; border: 0 !important; }
#diagnosis-sheet p, #diagnosis-sheet li { color: #263244 !important; line-height: 1.65; font-size: .88rem; }
#diagnosis-sheet ul, #diagnosis-sheet ol { margin: 4px 0 0 1.15rem; padding: 0; }
#diagnosis-sheet .info-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 4px 14px; }
#diagnosis-sheet .info-item {
  display: grid;
  grid-template-columns: 84px minmax(0,1fr);
  gap: 5px;
  min-width: 0;
  font-size: .82rem;
  padding: 1px 0;
}
#diagnosis-sheet .info-label {
  color: #596579 !important;
  font-weight: 800;
}
#diagnosis-sheet .info-value {
  min-width: 0 !important;
  max-width: 100% !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-all !important;
}
#diagnosis-sheet .warning-note { background: #fff8df !important; border: 1px solid #f3d66c !important; border-left: 4px solid #e2a800 !important; border-radius: 7px !important; padding: 9px 11px; color: #5d4900 !important; }
#diagnosis-sheet .clinical-reference { display: grid; gap: 10px; }
#diagnosis-sheet .clinical-reference-intro { margin: 0 !important; }
#diagnosis-sheet .clinical-reference-item { border-left: 3px solid var(--teal) !important; padding: 7px 10px; background: #f8fffc !important; border-radius: 5px !important; }
#diagnosis-sheet .clinical-reference-item strong { display: block; color: #164e63 !important; font-size: .92rem; margin-bottom: 3px; }
#diagnosis-sheet .clinical-reference-item p { margin: 0 !important; }
#diagnosis-sheet .clinical-reference-empty { background: #f7f9fb !important; border-radius: 6px !important; padding: 9px 11px; color: #445065 !important; }
#diagnosis-sheet .clinical-reference-warning { background: #fff8df !important; border: 1px solid #f3d66c !important; border-left: 4px solid #e2a800 !important; border-radius: 7px !important; padding: 9px 11px; color: #5d4900 !important; font-weight: 800; }
#diagnosis-sheet .sheet-downloads { background: #f7f9fb !important; border: 1px solid #e0e5eb !important; border-radius: 7px !important; margin-top: 10px; padding: 8px 10px; }

.system-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px 24px; font-size: .84rem; }
.system-item { display: grid; grid-template-columns: 158px minmax(0,1fr); gap: 8px; }
.system-label { font-weight: 900; color: #303b4d !important; }
.system-value-ok { color: var(--green) !important; font-weight: 900; }

@media (max-width: 1050px) {
  .app-shell { width: min(100% - 20px, 920px); }
  .main-grid {
    flex-direction: column !important;
    flex-wrap: nowrap !important;
  }
  .left-pane,
  .right-pane {
    width: 100% !important;
    flex: 1 1 auto !important;
  }
}
@media (max-width: 720px) {
  .app-topbar { flex-direction: column; }
  .top-health { width: 100%; }
  #diagnosis-sheet .report-top-grid { grid-template-columns: 1fr; }
  #diagnosis-sheet .basic-info-box { border-right: 0 !important; border-bottom: 1px solid #e3e7ed !important; padding: 0 0 10px; }
  .system-grid { grid-template-columns: 1fr; }
}

@page { size: A4; margin: 10mm; }
@media print {
  body, .gradio-container { background: #fff !important; }
  .app-topbar, .patient-card, .left-pane, .summary-card, .system-card, .report-toolbar, footer, .no-print { display: none !important; }
  .app-shell, .main-grid, .right-pane, .report-card, .report-body { background: #fff !important; border: 0 !important; box-shadow: none !important; margin: 0 !important; max-width: 100% !important; padding: 0 !important; width: 100% !important; }
  #diagnosis-sheet { border: 0 !important; box-shadow: none !important; margin: 0 !important; padding: 0 !important; width: 100% !important; }
  #diagnosis-sheet .sheet-downloads, #diagnosis-sheet .no-print { display: none !important; }
}
"""
SESSION_CSV_FIELDS = [
    "timestamp",
    "image_filename",
    "image_sha256",
    "probability_aortic_enlargement",
    "probability_cardiomegaly",
    "probability_pleural_thickening",
    "probability_pulmonary_fibrosis",
    "probability_pleural_effusion",
    "threshold_aortic_enlargement",
    "threshold_cardiomegaly",
    "threshold_pleural_thickening",
    "threshold_pulmonary_fibrosis",
    "threshold_pleural_effusion",
    "predicted_label_vector",
    "predicted_class_ids",
    "predicted_class_names",
    "classification_seconds",
    "device",
    "model_sha256",
    "threshold_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_path(project_root: Path, value: Path) -> Path:
    expanded = value.expanduser()
    return (expanded if expanded.is_absolute() else project_root / expanded).resolve()


def dependency_versions() -> dict[str, str]:
    packages = ["gradio", "numpy", "pandas", "Pillow", "torch", "torchvision"]
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def dependency_audit() -> dict[str, dict[str, Any]]:
    import_names = {
        "gradio": "gradio",
        "numpy": "numpy",
        "pandas": "pandas",
        "Pillow": "PIL",
        "torch": "torch",
        "torchvision": "torchvision",
    }
    versions = dependency_versions()
    audit: dict[str, dict[str, Any]] = {}
    for package, module_name in import_names.items():
        try:
            __import__(module_name)
            importable = True
            error = None
        except Exception as exc:
            importable = False
            error = f"{type(exc).__name__}: {exc}"
        audit[package] = {
            "version": versions.get(package, "UNKNOWN"),
            "importable": importable,
            "status": "PASS" if importable and versions.get(package) != "NOT_INSTALLED" else "FAIL",
            "error": error,
        }
    return audit


def display_path(value: Any, project_root: Path = PROJECT_DEFAULT) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        path_text = str(value)
        path_like = True
    else:
        path_text = str(value)
        path_like = (
            bool(re.match(r"^[A-Za-z]:[\\/]", path_text))
            or path_text.startswith(("/", "\\", "./", "../", ".\\", "..\\"))
            or "\\" in path_text
            or path_text.startswith(("outputs/", "data/", "docs/", "src/", "scripts/"))
        )
    if not path_text or path_text.startswith(("http://", "https://")) or not path_like:
        return path_text
    try:
        path = Path(path_text)
        resolved = path.expanduser().resolve()
        root = project_root.expanduser().resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            if "C:\\Users\\" in str(resolved):
                return f"<LOCAL_PATH_REDACTED>/{resolved.name}"
            return str(resolved)
    except Exception:
        if "C:\\Users\\" in path_text:
            return "<LOCAL_PATH_REDACTED>"
        return path_text


def sanitize_for_audit(payload: Any, project_root: Path) -> Any:
    if isinstance(payload, dict):
        return {key: sanitize_for_audit(value, project_root) for key, value in payload.items()}
    if isinstance(payload, list):
        return [sanitize_for_audit(value, project_root) for value in payload]
    if isinstance(payload, tuple):
        return [sanitize_for_audit(value, project_root) for value in payload]
    if isinstance(payload, Path):
        return display_path(payload, project_root)
    if isinstance(payload, str):
        return display_path(payload, project_root)
    return payload


def cuda_audit(device: torch.device) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def find_port_pids(port: int) -> list[int]:
    try:
        import psutil

        return sorted(
            {
                connection.pid
                for connection in psutil.net_connections(kind="tcp")
                if connection.pid
                and connection.laddr
                and connection.laddr.port == port
                and connection.status == psutil.CONN_LISTEN
            }
        )
    except Exception:
        return []


def port_statuses(host: str, ports: tuple[int, ...] = (7860, 7861)) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for port in ports:
        available = port_available(host, port)
        statuses[str(port)] = {
            "available": available,
            "occupied": not available,
            "occupied_pids": [] if available else find_port_pids(port),
        }
    return statuses


def select_port(host: str, requested: int) -> tuple[int, dict[str, Any]]:
    if port_available(host, requested):
        return requested, {
            "requested_port_occupied": False,
            "occupied_pids": [],
            "fallback_used": False,
        }
    pids = find_port_pids(requested)
    if requested != 7860:
        raise RuntimeError(f"Port {requested} is occupied; PID(s): {pids}")
    if not port_available(host, 7861):
        raise RuntimeError(f"Ports 7860 and 7861 are occupied; 7860 PID(s): {pids}")
    return 7861, {
        "requested_port_occupied": True,
        "occupied_pids": pids,
        "fallback_used": True,
        "fallback_port": 7861,
    }


def verify_port_failover(host: str) -> dict[str, Any]:
    if not port_available(host, 7860):
        selected, state = select_port(host, 7860)
        return {"status": "PASS", "mode": "already_occupied", "selected_port": selected, **state}
    if not port_available(host, 7861):
        return {
            "status": "SKIP",
            "reason": "Port 7861 is already occupied, so fallback simulation cannot reserve it safely.",
        }
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, 7860))
        probe.listen(1)
        selected, state = select_port(host, 7860)
        if selected != 7861 or not state.get("fallback_used"):
            raise RuntimeError("Port fallback did not select 7861 while 7860 was reserved")
        return {"status": "PASS", "mode": "simulated_7860_occupied", "selected_port": selected, **state}
    finally:
        probe.close()


def lan_ip_address() -> str:
    try:
        candidates = socket.gethostbyname_ex(socket.gethostname())[2]
    except Exception:
        candidates = []
    for candidate in candidates:
        if candidate and not candidate.startswith("127."):
            return candidate
    return "127.0.0.1"


def launch_urls(server_name: str, selected_port: int, lan: bool) -> dict[str, Any]:
    urls = {
        "local_url": f"http://127.0.0.1:{selected_port}",
        "public_share_url": None,
        "gradio_live_url_created": False,
    }
    if lan:
        urls["lan_url"] = f"http://{lan_ip_address()}:{selected_port}"
        urls["lan_warning"] = "LAN mode exposes the local demo to devices on the same trusted network only."
    else:
        urls["lan_url"] = None
    urls["server_bind"] = f"{server_name}:{selected_port}"
    return urls


def check_output_write_permission(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=".write_test_", suffix=".tmp", dir=output_dir
    )
    os.close(handle)
    Path(temporary_name).unlink()


def patient_info(
    record_number: str,
    name: str,
    sex: str,
    age: Any,
    exam_date: str,
    note: str,
) -> dict[str, Any]:
    return {
        "record_number": record_number or "",
        "name": name or "",
        "sex": sex or "",
        "age": "" if age is None else age,
        "exam_date": exam_date or "",
        "note": note or "",
    }


def prediction_csv_row(prediction: dict[str, Any]) -> dict[str, Any]:
    slugs = (
        "aortic_enlargement",
        "cardiomegaly",
        "pleural_thickening",
        "pulmonary_fibrosis",
        "pleural_effusion",
    )
    row = {
        "timestamp": prediction["timestamp"],
        "image_filename": prediction["image_filename"],
        "image_sha256": prediction["image_sha256"],
        "predicted_label_vector": json.dumps(
            prediction["predicted_label_vector"], separators=(",", ":")
        ),
        "predicted_class_ids": "|".join(map(str, prediction["predicted_class_ids"])),
        "predicted_class_names": "|".join(prediction["predicted_class_names_en"]),
        "classification_seconds": f"{prediction['classification_seconds']:.8f}",
        "device": prediction["device"],
        "model_sha256": prediction["model_sha256"],
        "threshold_sha256": prediction["threshold_sha256"],
    }
    for index, slug in enumerate(slugs):
        row[f"probability_{slug}"] = f"{prediction['class_probabilities'][index]:.8f}"
        row[f"threshold_{slug}"] = f"{prediction['class_thresholds'][index]:.8f}"
    return row


def pure_model_prediction(prediction: dict[str, Any], project_root: Path = PROJECT_DEFAULT) -> dict[str, Any]:
    keys = [
        "timestamp",
        "model_path",
        "model_sha256",
        "threshold_path",
        "threshold_sha256",
        "architecture",
        "initialization",
        "device",
        "preprocessing_description",
        "input_tensor_shape",
        "logits_shape",
        "probabilities_shape",
        "predicted_vector_shape",
        "class_probabilities",
        "class_thresholds",
        "probability_rows",
        "predicted_label_vector",
        "predicted_class_ids",
        "predicted_class_names_en",
        "predicted_class_names_zh",
        "no_positive_message",
        "preprocessing_seconds",
        "classification_seconds",
        "total_model_seconds",
        "uses_bbox",
        "uses_roi_crop",
        "uses_yolo",
        "uses_softmax",
        "uses_sigmoid",
        "optimizer_created",
        "backward_executed",
        "test_images_read_count",
        "disclaimer",
    ]
    return sanitize_for_audit({key: prediction[key] for key in keys}, project_root)


def ground_truth_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    if not prediction["ground_truth_catalog_match"]:
        return {
            "catalog_match": False,
            "message": GROUND_TRUTH_UNAVAILABLE,
            "ground_truth_label_vector": None,
            "accuracy_computed": False,
        }
    keys = [
        "ground_truth_catalog_match",
        "ground_truth_match_method",
        "ground_truth_image_id",
        "ground_truth_label_vector",
        "ground_truth_class_ids",
        "ground_truth_class_names_en",
        "ground_truth_class_names_zh",
        "correctly_detected_class_ids",
        "correctly_detected_labels",
        "missed_class_ids",
        "missed_labels",
        "extra_class_ids",
        "extra_predicted_labels",
        "tp",
        "fp",
        "fn",
        "tn",
        "exact_match",
        "sample_precision",
        "sample_recall",
        "sample_f1",
    ]
    payload = {key: prediction[key] for key in keys}
    payload["catalog_match"] = True
    payload["accuracy_computed"] = True
    return payload


def generate_report_safe(
    ollama: FullImageMultilabelOllamaService,
    structured_result: dict[str, Any],
    patient: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Generate only the auxiliary narrative; fixed clinical references are separate."""
    try:
        result = ollama.generate(structured_result, patient)
        response = ensure_disclaimer(result["response"])
        result = {**result, "response": response}
        return response, result
    except Exception as exc:
        fallback = ensure_disclaimer(OLLAMA_FALLBACK)
        return fallback, {
            "status": "UNAVAILABLE",
            "backend": ollama.base_url,
            "model": ollama.selected_model or ollama.requested_model,
            "generation_seconds": 0.0,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "image_sent_to_ollama": False,
            "ground_truth_sent_to_ollama": False,
            "response": fallback,
        }



CLINICAL_REFERENCE_INTRO = (
    "以下內容依據本次影像分析結果整理，供臨床評估參考；"
    "實際判斷仍應結合病史、症狀、理學檢查、既往影像及正式放射科判讀。"
)
CLINICAL_REFERENCE_NO_POSITIVE = (
    "目前未出現超過判定門檻的類別，"
    "建議仍依臨床表現與正式影像判讀綜合評估。"
)
CLINICAL_REFERENCE_WARNING = (
    "本系統為研究型 AI 輔助工具，本區內容不得單獨作為"
    "診斷、檢查或治療決策依據。"
)

# 五類臨床評估文字為設計者依一般國際臨床指引背景整理的固定研究型參考內容，不由 Ollama 自由生成。
CLINICAL_REFERENCE_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "Aortic enlargement",
        "主動脈擴大",
        "建議比對既往影像，並結合病史、血壓及正式影像判讀，"
        "進一步評估主動脈輪廓與尺寸；是否需要其他影像檢查，"
        "應依個別臨床情況決定。",
    ),
    (
        "Cardiomegaly",
        "心臟擴大",
        "建議結合症狀、病史、既往影像及正式影像判讀，"
        "進一步評估心臟結構與功能；必要時可依臨床情況"
        "考慮心電圖或心臟超音波評估。",
    ),
    (
        "Pleural thickening",
        "胸膜（肋膜）增厚",
        "建議比對既往影像；若合併不明原因的單側胸腔積液"
        "或其他可疑影像特徵，可依臨床情況評估追蹤影像"
        "或進一步胸部 CT。",
    ),
    (
        "Pulmonary fibrosis",
        "肺纖維化",
        "建議結合肺功能、既往影像及臨床表現，"
        "進一步評估間質性肺部疾病的可能性；"
        "必要時可依臨床情況考慮高解析度胸部 CT"
        "或相關專科評估。",
    ),
    (
        "Pleural effusion",
        "胸膜（肋膜）積液",
        "建議結合症狀、病史、積液側別、範圍及正式影像判讀，"
        "進一步評估可能病因；可依臨床情況考慮追蹤影像、"
        "胸腔超音波或胸部 CT。",
    ),
)


def predicted_label_names(payload: dict[str, Any] | None) -> set[str]:
    if not payload:
        return set()
    names = {str(name) for name in (payload.get("predicted_class_names_en") or [])}
    if names:
        return names
    vector = payload.get("predicted_label_vector") or payload.get("predicted_vector")
    if vector:
        for index, value in enumerate(vector):
            if value:
                mapped = CLASS_MAPPING_EN.get(index)
                if mapped:
                    names.add(mapped)
    return names


def clinical_reference_entries(payload: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    """Return fixed entries in the formal five-class model order."""
    predicted = predicted_label_names(payload)
    return [rule for rule in CLINICAL_REFERENCE_RULES if rule[0] in predicted]


def build_followup_guidance(payload: dict[str, Any] | None) -> str:
    """Build deterministic clinical reference Markdown from thresholded predictions."""
    sections = ["## 臨床評估參考", CLINICAL_REFERENCE_INTRO]
    entries = clinical_reference_entries(payload)
    if entries:
        for _model_label, display_name, fixed_text in entries:
            sections.append(f"**{display_name}**\n{fixed_text}")
    else:
        sections.append(CLINICAL_REFERENCE_NO_POSITIVE)
    sections.append(CLINICAL_REFERENCE_WARNING)
    return "\n\n".join(sections)

def ensure_disclaimer(report: str) -> str:
    if DISCLAIMER in report:
        return report
    return f"{report.rstrip()}\n\n---\n{DISCLAIMER}"


def markdown_to_plain_text(markdown: str) -> str:
    text_value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)
    text_value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text_value)
    text_value = re.sub(r"^#{1,6}\s*", "", text_value, flags=re.MULTILINE)
    text_value = text_value.replace("**", "").replace("__", "").replace("`", "")
    text_value = text_value.replace("---", "")
    return text_value.strip()


def report_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_names = [
        r"C:\Windows\Fonts\msjhbd.ttc" if bold else r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",
        r"C:\Windows\Fonts\NotoSansTC-Regular.otf",
        r"C:\Windows\Fonts\Arial.ttf",
    ]
    for font_name in font_names:
        try:
            if Path(font_name).is_file():
                return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrapped_pdf_lines(draw: ImageDraw.ImageDraw, text_value: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text_value.splitlines():
        paragraph = paragraph.rstrip()
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def write_report_pdf(path: Path, markdown: str) -> None:
    width, height = 1240, 1754
    margin_x, margin_y = 90, 92
    title_font = report_font(34, bold=True)
    body_font = report_font(24)
    small_font = report_font(20)
    line_height = 36
    title = "AI \u8f14\u52a9\u80f8\u8154 X \u5149\u591a\u6a19\u7c64\u8fa8\u8b58\u8aaa\u660e\u66f8"
    plain_text = markdown_to_plain_text(markdown)
    pages: list[Image.Image] = []

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        page = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(page)
        draw.text((margin_x, margin_y), title, fill=(17, 24, 39), font=title_font)
        draw.line((margin_x, margin_y + 52, width - margin_x, margin_y + 52), fill=(20, 184, 166), width=4)
        return page, draw, margin_y + 86

    page, draw, y = new_page()
    for line in wrapped_pdf_lines(draw, plain_text, body_font, width - 2 * margin_x):
        if y + line_height > height - margin_y:
            draw.text((margin_x, height - margin_y + 25), DISCLAIMER, fill=(180, 83, 9), font=small_font)
            pages.append(page)
            page, draw, y = new_page()
        draw.text((margin_x, y), line, fill=(17, 24, 39), font=body_font)
        y += line_height if line else int(line_height * 0.7)
    draw.text((margin_x, height - margin_y + 25), DISCLAIMER, fill=(180, 83, 9), font=small_font)
    pages.append(page)
    temporary = path.with_name(path.name + ".writing")
    pages[0].save(temporary, "PDF", resolution=150.0, save_all=True, append_images=pages[1:])
    os.replace(temporary, path)


def html_text(value: Any) -> str:
    text_value = "" if value is None else str(value)
    return html.escape(text_value, quote=True)


def paragraph_html(value: str) -> str:
    escaped = html_text(value).replace("\n", "<br>")
    return escaped or "\u672a\u63d0\u4f9b"


def patient_display_value(patient: dict[str, Any], key: str) -> str:
    value = patient.get(key)
    if value is None or value == "":
        return "\u672a\u63d0\u4f9b"
    return str(value)


def predicted_pairs(prediction: dict[str, Any] | None) -> list[str]:
    if not prediction:
        return []
    names_en = prediction.get("predicted_class_names_en") or []
    names_zh = prediction.get("predicted_class_names_zh") or []
    return [f"{zh}\uff08{en}\uff09" for en, zh in zip(names_en, names_zh)]


def report_payload_from_prediction(prediction: dict[str, Any] | None) -> dict[str, Any]:
    if not prediction:
        return {"predicted_class_names_en": [], "predicted_label_vector": [0, 0, 0, 0, 0]}
    return {
        "predicted_class_names_en": list(prediction.get("predicted_class_names_en") or []),
        "predicted_label_vector": list(prediction.get("predicted_label_vector") or [0, 0, 0, 0, 0]),
    }


def predicted_positive_sentence(prediction: dict[str, Any] | None) -> str:
    labels = predicted_pairs(prediction)
    if not labels:
        return "\u6a21\u578b\u672a\u5075\u6e2c\u5230\u660e\u986f\u9054\u5230\u5224\u5b9a\u689d\u4ef6\u7684\u967d\u6027\u985e\u5225\uff1b\u82e5\u4ecd\u6709\u81e8\u5e8a\u7591\u616e\uff0c\u4ecd\u9700\u7531\u91ab\u5e2b\u78ba\u8a8d\u3002"
    return "\u6a21\u578b\u9810\u6e2c\u6b64\u80f8\u8154 X \u5149\u5f71\u50cf\u53ef\u80fd\u5448\u73fe\u8207\u300c" + "\u3001".join(labels) + "\u300d\u76f8\u95dc\u7684\u5f71\u50cf\u7279\u5fb5\u3002"


def markdown_section(markdown: str, heading: str) -> str:
    if heading not in markdown:
        return ""
    start_index = markdown.index(heading) + len(heading)
    next_index = markdown.find("\n## ", start_index)
    section = markdown[start_index:] if next_index == -1 else markdown[start_index:next_index]
    return markdown_to_plain_text(section).strip()


def remove_technical_report_lines(text_value: str) -> str:
    forbidden_terms = (
        "probability",
        "Validation threshold",
        "threshold",
        "Predicted vector",
        "ConvNeXt",
        "ImageNet",
        "normalization",
        "state_dict",
        "checkpoint",
        "Ground Truth",
        "TP/FP/FN/TN",
        "TP / FP / FN",
        "BBox",
        "ROI",
        "224x224",
        "224\u00d7224",
        "224\\u00d7224",
    )
    kept = []
    for line in text_value.splitlines():
        if any(term in line for term in forbidden_terms):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def report_findings_text(report: str, prediction: dict[str, Any] | None) -> str:
    headings = (
        "## 影像分析所見",
        "## Findings",
        "## 放射科影像報告",
        "## 影像所見",
    )
    sections = [markdown_section(report, heading) for heading in headings]
    text_value = next((section for section in sections if section), "")
    if not text_value:
        text_value = predicted_positive_sentence(prediction)
    text_value = remove_technical_report_lines(text_value)
    return text_value or predicted_positive_sentence(prediction)


def clinical_diagnosis_html(prediction: dict[str, Any] | None) -> str:
    labels = predicted_pairs(prediction)
    if not labels:
        return (
            "<p>本模型於五個目標類別中未形成陽性判定；"
            "此結果不代表胸腔 X 光正常或沒有其他異常。</p>"
        )
    items = "".join(f"<li>{html_text(label)}</li>" for label in labels)
    return f"<ul class=\"clinical-diagnosis-list\">{items}</ul>"


def clinical_diagnosis_markdown(prediction: dict[str, Any] | None) -> list[str]:
    labels = predicted_pairs(prediction)
    if not labels:
        return ["本模型於五個目標類別中未形成陽性判定；此結果不代表胸腔 X 光正常或沒有其他異常。"]
    return [f"- {label}" for label in labels]

def report_impression_items(prediction: dict[str, Any] | None) -> list[str]:
    items = [predicted_positive_sentence(prediction)]
    if predicted_pairs(prediction):
        items.append("\u4ee5\u4e0a\u7d50\u679c\u4ee3\u8868\u6a21\u578b\u5075\u6e2c\u5230\u8207\u8a72\u985e\u5225\u76f8\u95dc\u7684\u5f71\u50cf\u7279\u5fb5\uff0c\u4e0d\u4ee3\u8868\u6b63\u5f0f\u8a3a\u65b7\u3002")
    items.append("\u5efa\u8b70\u7531\u91ab\u5e2b\u7d50\u5408\u539f\u59cb\u5f71\u50cf\u3001\u75c7\u72c0\u3001\u75c5\u53f2\u8207\u5176\u4ed6\u6aa2\u67e5\u7d50\u679c\u9032\u4e00\u6b65\u78ba\u8a8d\u3002")
    return items


def clean_recommendation_text(prediction: dict[str, Any] | None) -> str:
    """Return the fixed clinical reference section without its Markdown heading."""
    recommendation = build_followup_guidance(report_payload_from_prediction(prediction))
    return recommendation.removeprefix("## 臨床評估參考\n\n").strip()


def clinical_reference_html(prediction: dict[str, Any] | None) -> str:
    """Render fixed clinical reference content without using Ollama or Ground Truth."""
    entries = clinical_reference_entries(report_payload_from_prediction(prediction))
    if entries:
        content = "".join(
            "<div class='clinical-reference-item'>"
            f"<strong>{html_text(display_name)}</strong>"
            f"<p>{paragraph_html(fixed_text)}</p>"
            "</div>"
            for _model_label, display_name, fixed_text in entries
        )
    else:
        content = (
            "<div class='clinical-reference-empty'>"
            f"{paragraph_html(CLINICAL_REFERENCE_NO_POSITIVE)}"
            "</div>"
        )
    return (
        "<div class='clinical-reference'>"
        f"<p class='clinical-reference-intro'>{paragraph_html(CLINICAL_REFERENCE_INTRO)}</p>"
        f"{content}"
        "<div class='clinical-reference-warning'>"
        f"{paragraph_html(CLINICAL_REFERENCE_WARNING)}"
        "</div></div>"
    )


def validate_clinical_reference_rules() -> None:
    """Fail fast if the five deterministic rule mappings drift or leak other inputs."""
    expected_labels = [rule[0] for rule in CLINICAL_REFERENCE_RULES]
    expected_names = [rule[1] for rule in CLINICAL_REFERENCE_RULES]
    if expected_labels != [
        "Aortic enlargement",
        "Cardiomegaly",
        "Pleural thickening",
        "Pulmonary fibrosis",
        "Pleural effusion",
    ]:
        raise RuntimeError("Clinical reference model-label order changed")

    all_positive = {
        "predicted_class_names_en": expected_labels,
        "predicted_label_vector": [1, 1, 1, 1, 1],
    }
    all_text = build_followup_guidance(all_positive)
    positions = [all_text.index(name) for name in expected_names]
    if positions != sorted(positions):
        raise RuntimeError("Clinical reference display order changed")
    if any(all_text.count(name) != 1 for name in expected_names):
        raise RuntimeError("All-positive clinical reference test failed")

    for index, (model_label, display_name, _fixed_text) in enumerate(CLINICAL_REFERENCE_RULES):
        vector = [0, 0, 0, 0, 0]
        vector[index] = 1
        single = build_followup_guidance({
            "predicted_class_names_en": [model_label],
            "predicted_label_vector": vector,
        })
        if single.count(display_name) != 1:
            raise RuntimeError(f"Single-positive clinical reference test failed: {model_label}")
        if any(other in single for other in expected_names if other != display_name):
            raise RuntimeError(f"Single-positive isolation failed: {model_label}")

    none_text = build_followup_guidance({
        "predicted_class_names_en": [],
        "predicted_label_vector": [0, 0, 0, 0, 0],
    })
    if CLINICAL_REFERENCE_NO_POSITIVE not in none_text:
        raise RuntimeError("All-negative clinical reference test failed")

    prediction = {
        "predicted_class_names_en": ["Cardiomegaly"],
        "predicted_label_vector": [0, 1, 0, 0, 0],
    }
    with_ground_truth = {
        **prediction,
        "ground_truth_label_vector": [1, 0, 1, 0, 1],
        "exact_match": False,
    }
    if clinical_reference_html(prediction) != clinical_reference_html(with_ground_truth):
        raise RuntimeError("Ground Truth affected clinical reference content")

    forbidden = ("BTS", "NICE", "ACC", "AHA", "source", "rule_id")
    rendered = clinical_reference_html(all_positive)
    if any(term in rendered for term in forbidden):
        raise RuntimeError("Forbidden front-end term found in clinical reference content")


def report_download_links_html(session_dir: Path | None) -> str:
    if session_dir is None:
        return ""
    links = []
    pdf_path = session_dir / "diagnosis_report.pdf"
    md_path = session_dir / "diagnosis_report.md"
    if pdf_path.is_file():
        links.append(f"<a href=\"{html_text(gradio_file_url(pdf_path))}\" download>\u4e0b\u8f09 PDF \u8aaa\u660e\u66f8</a>")
    if md_path.is_file():
        links.append(f"<a href=\"{html_text(gradio_file_url(md_path))}\" download>\u4e0b\u8f09 Markdown \u8aaa\u660e\u66f8</a>")
    if not links:
        return ""
    return "<div class=\"sheet-downloads no-print\"><strong>\u5831\u544a\u4e0b\u8f09\uff1a</strong> " + " \u00a0|\u00a0 ".join(links) + "</div>"


def blank_report_sheet_html(message: str | None = None) -> str:
    message = message or "請先上傳完整胸腔 X 光影像並按下「開始分析」，完成後本區會顯示可列印的診斷說明書。"
    return (
        "<article id=\"diagnosis-sheet\" class=\"diagnosis-sheet\">"
        "<div class=\"sheet-header\"><div>"
        "<h2>AI 輔助胸腔 X 光診斷說明書</h2>"
        "<div class=\"sheet-subtitle\">AI-assisted Chest X-ray Diagnostic Report</div>"
        "</div><div class=\"research-stamp\">研究驗證</div></div>"
        f"<section class=\"sheet-section\"><p>{paragraph_html(message)}</p></section>"
        "</article>"
    )


def report_sheet_html(report: str, prediction: dict[str, Any] | None, patient: dict[str, Any], session_dir: Path | None) -> str:
    image_filename = "未提供" if not prediction else str(prediction.get("image_filename") or "未提供")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_rows = [
        ("病歷號碼", patient_display_value(patient, "record_number")),
        ("姓名", patient_display_value(patient, "name")),
        ("性別", patient_display_value(patient, "sex")),
        ("年齡", patient_display_value(patient, "age")),
        ("檢查日期", patient_display_value(patient, "exam_date")),
        ("影像編號", image_filename),
    ]
    patient_html = "".join(
        f"<div class=\"info-item\"><span class=\"info-label\">{html_text(label)}：</span>"
        f"<span class=\"info-value\">{html_text(value)}</span></div>"
        for label, value in patient_rows
    )
    findings = report_findings_text(report, prediction)
    impression = "".join(f"<li>{paragraph_html(item)}</li>" for item in report_impression_items(prediction))
    clinical_reference = clinical_reference_html(prediction)
    return (
        "<article id=\"diagnosis-sheet\" class=\"diagnosis-sheet\">"
        "<div class=\"sheet-header\"><div>"
        "<h2>AI 輔助胸腔 X 光診斷說明書</h2>"
        "<div class=\"sheet-subtitle\">AI-assisted Chest X-ray Diagnostic Report</div>"
        f"<div class=\"sheet-meta\">報告產生時間：{html_text(generated_at)}</div>"
        "</div><div class=\"research-stamp\">研究驗證</div></div>"
        "<div class=\"report-top-grid\">"
        "<section class=\"basic-info-box\"><h3>基本資料</h3>"
        f"<div class=\"info-grid\">{patient_html}</div></section>"
        "<section class=\"clinical-box\"><h3>臨床診斷（模型達門檻分類）</h3>"
        f"{clinical_diagnosis_html(prediction)}</section>"
        "</div>"
        "<section class=\"sheet-section\"><h3>◉ 影像分析所見</h3>"
        f"<p>{paragraph_html(findings)}</p></section>"
        "<section class=\"sheet-section\"><h3>◉ 影像印象</h3>"
        f"<ol>{impression}</ol></section>"
        "<section class=\"sheet-section\"><h3>◉ 臨床評估參考</h3>"
        f"{clinical_reference}</section>"
        "</article>"
    )


def report_sheet_markdown(report: str, prediction: dict[str, Any] | None, patient: dict[str, Any]) -> str:
    image_filename = "未提供" if not prediction else str(prediction.get("image_filename") or "未提供")
    lines = [
        "# AI 輔助胸腔 X 光診斷說明書",
        "",
        "用途：研究與教學展示／非正式診斷",
        f"報告產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 基本資料",
        f"- 姓名：{patient_display_value(patient, 'name')}",
        f"- 性別：{patient_display_value(patient, 'sex')}",
        f"- 年齡：{patient_display_value(patient, 'age')}",
        f"- 病歷號碼：{patient_display_value(patient, 'record_number')}",
        f"- 檢查日期：{patient_display_value(patient, 'exam_date')}",
        f"- 影像檔名：{image_filename}",
        "",
        "## 2. 臨床診斷",
    ]
    lines.extend(clinical_diagnosis_markdown(prediction))
    lines.extend([
        "",
        "## 3. 影像分析所見（Findings）",
        report_findings_text(report, prediction),
        "",
        "## 4. 影像印象（Impression）",
    ])
    lines.extend(f"{index}. {item}" for index, item in enumerate(report_impression_items(prediction), start=1))
    lines.extend([
        "",
        "## 5. 臨床評估參考",
        clean_recommendation_text(prediction),
        "",
    ])
    return "\n".join(lines)

def downloadable_report_markdown(prediction: dict[str, Any] | None, patient: dict[str, Any], report: str) -> str:
    return report_sheet_markdown(report, prediction, patient)


def write_report_artifacts(directory: Path, prediction: dict[str, Any] | None, patient: dict[str, Any], report: str) -> None:
    markdown = downloadable_report_markdown(prediction, patient, report)
    atomic_write_text(directory / "diagnosis_report.md", markdown.rstrip() + "\n")
    try:
        write_report_pdf(directory / "diagnosis_report.pdf", markdown)
    except Exception as exc:
        atomic_write_text(
            directory / "diagnosis_report_pdf_error.txt",
            f"PDF generation failed: {type(exc).__name__}: {exc}\n",
        )


def gradio_file_url(path: Path) -> str:
    return "/file=" + quote(path.resolve().as_posix(), safe="/:")


def report_with_download_links(report: str, session_dir: Path | None) -> str:
    if session_dir is None:
        return report
    return report + "\n"
def persist_session(
    output_dir: Path,
    prediction: dict[str, Any],
    ollama_payload: dict[str, Any],
    report: str,
    ollama_audit: dict[str, Any],
    patient: dict[str, Any],
    total_seconds: float,
    smoke_test: bool,
) -> Path:
    sessions_dir = output_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_name = f"{timestamp_slug()}_{uuid.uuid4().hex[:8]}"
    session_dir = sessions_dir / session_name
    staging = sessions_dir / f"{session_name}.writing"
    if session_dir.exists() or staging.exists():
        raise RuntimeError("Session path collision; refusing to overwrite")
    staging.mkdir()
    try:
        input_metadata = {
            key: prediction[key]
            for key in (
                "timestamp",
                "image_filename",
                "image_path",
                "image_sha256",
                "image_sha256_short",
                "original_width",
                "original_height",
                "original_mode",
                "model_input_width",
                "model_input_height",
                "source_image_unchanged",
            )
        }
        atomic_write_json(staging / "input_metadata.json", sanitize_for_audit(input_metadata, PROJECT_DEFAULT))
        atomic_write_json(staging / "model_prediction.json", pure_model_prediction(prediction, PROJECT_DEFAULT))
        row = prediction_csv_row(prediction)
        atomic_write_csv(staging / "model_prediction.csv", SESSION_CSV_FIELDS, [row])
        atomic_write_json(staging / "ground_truth_comparison.json", ground_truth_payload(prediction))
        atomic_write_text(staging / "ollama_report.txt", report.rstrip() + "\n")
        write_report_artifacts(staging, prediction, patient, report)
        ollama_audit_clean = {key: value for key, value in ollama_audit.items() if key != "response"}
        ollama_audit_clean.update({
            "structured_input_fields": sorted(ollama_payload),
            "ground_truth_sent_to_ollama": False,
            "image_sent_to_ollama": False,
        })
        atomic_write_json(staging / "ollama_audit.json", ollama_audit_clean)
        session_audit = {
            "status": "PASS",
            "timestamp": utc_now(),
            "smoke_test": smoke_test,
            "classification_completed": True,
            "ollama_status": ollama_audit.get("status"),
            "ollama_failure_did_not_invalidate_classification": True,
            "model_inference_count_this_session": 1,
            "test_images_read_count": 0,
            "uses_full_image": True,
            "uses_bbox": False,
            "uses_roi_crop": False,
            "uses_yolo": False,
            "uses_softmax": False,
            "uses_sigmoid": True,
            "optimizer_created": False,
            "backward_executed": False,
            "ground_truth_sent_to_ollama": False,
            "image_sent_to_ollama": False,
            "patient_information_logged": False,
            "classification_seconds": prediction["classification_seconds"],
            "ollama_generation_seconds": ollama_audit.get("generation_seconds", 0.0),
            "total_processing_seconds": total_seconds,
            "research_disclaimer_present": DISCLAIMER in report or report == OLLAMA_FALLBACK,
            "source_image_unchanged": prediction["source_image_unchanged"],
        }
        atomic_write_json(staging / "session_audit.json", session_audit)
        staging.rename(session_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    remnants = [
        str(path)
        for path in session_dir.rglob("*")
        if path.name.endswith((".tmp", ".writing"))
    ]
    if remnants:
        raise RuntimeError(f"Temporary session remnants found: {remnants}")
    return session_dir


def format_prediction_summary(prediction: dict[str, Any]) -> str:
    if prediction["predicted_class_ids"]:
        items = "\n".join(
            f"- {zh}（{en}）"
            for en, zh in zip(
                prediction["predicted_class_names_en"],
                prediction["predicted_class_names_zh"],
            )
        )
        return (
            "### 達門檻類別（依機率由高至低）\n"
            f"{items}\n\n"
            f"<div class='vector-note'><strong>Predicted Vector：</strong>"
            f"<code>{prediction['predicted_label_vector']}</code><br>"
            "1＝達門檻（陽性），0＝未達門檻（陰性）</div>"
        )
    return (
        "### 達門檻類別\n"
        f"{prediction['no_positive_message']}\n\n"
        f"<div class='vector-note'><strong>Predicted Vector：</strong>"
        f"<code>{prediction['predicted_label_vector']}</code></div>"
    )


def format_ground_truth(prediction: dict[str, Any], catalog_path: Path | str | None = None) -> str:
    catalog_display = str(catalog_path or DEFAULT_CATALOG_RELATIVE)
    source_filename = prediction.get("image_filename", "N/A")
    source_sha = prediction.get("image_sha256_short") or prediction.get("image_sha256") or "N/A"
    matched_image_id = prediction.get("ground_truth_image_id") or Path(str(source_filename)).stem
    match_method = prediction.get("ground_truth_match_method") or "not matched"

    header = (
        "### Ground Truth \u4f86\u6e90\u8207\u6bd4\u5c0d\u8aaa\u660e\n"
        f"- **\u4e0a\u50b3\u6a94\u6848\uff1a** `{source_filename}`\n"
        f"- **Matched image_id\uff1a** `{matched_image_id}`\n"
        f"- **\u6bd4\u5c0d\u65b9\u5f0f\uff1a** `{match_method}`\uff08\u7cfb\u7d71\u512a\u5148\u4f7f\u7528 SHA256\uff0c\u6bd4\u5c0d\u4e0d\u5230\u624d\u4f7f\u7528 image_id\uff09\n"
        f"- **\u5f71\u50cf SHA256\uff1a** `{source_sha}`\n"
        f"- **Ground Truth \u4f86\u6e90\uff1a** `{catalog_display}`\n"
        "- **\u7528\u9014\uff1a** \u50c5\u4f9b Demo \u9a57\u8b49\u6a21\u578b\u8f38\u51fa\u662f\u5426\u8207\u65e2\u6709\u6a19\u8a3b\u4e00\u81f4\uff0c\u4e0d\u53c3\u8207\u6a21\u578b\u8a13\u7df4\u3001\u8abf\u53c3\u6216\u6b63\u5f0f\u8a3a\u65b7\u3002\n"
    )
    if not prediction["ground_truth_catalog_match"]:
        return (
            header
            + "\n**Ground Truth \u67e5\u8a62\u7d50\u679c\uff1a** \u672a\u5728\u76ee\u524d Ground Truth Catalog / manifest \u627e\u5230\u5c0d\u61c9\u7d00\u9304\u3002  \n"
            + "\u56e0\u6b64\u672c\u6848\u4f8b\u53ea\u986f\u793a\u6a21\u578b\u9810\u6e2c\uff0c\u4e0d\u8a08\u7b97 TP / FP / FN \u6216 Exact Match\u3002"
        )

    gt = "\u3001".join(prediction["ground_truth_class_names_zh"] or []) or "\u7a7a\u96c6\u5408"
    predicted = "\u3001".join(prediction["predicted_class_names_zh"] or []) or "\u7121\u985e\u5225\u9054\u9580\u6abb"
    correct = "\u3001".join(prediction["correctly_detected_labels"] or []) or "\u7121"
    missed = "\u3001".join(prediction["missed_labels"] or []) or "\u7121"
    extra = "\u3001".join(prediction["extra_predicted_labels"] or []) or "\u7121"
    if prediction["exact_match"]:
        interpretation = "\u5b8c\u5168\u7b26\u5408 Ground Truth \u6a19\u8a3b\u3002"
    elif prediction["tp"] > 0:
        interpretation = "\u90e8\u5206\u547d\u4e2d Ground Truth\uff0c\u4f46\u4ecd\u5b58\u5728\u6f0f\u5224\u6216\u984d\u5916\u967d\u6027\u9810\u6e2c\u3002"
    else:
        interpretation = "\u672a\u547d\u4e2d\u76ee\u524d Ground Truth \u7684\u967d\u6027\u6a19\u8a3b\uff0c\u9069\u5408\u4f5c\u70ba\u932f\u8aa4\u6848\u4f8b\u5206\u6790\u3002"

    return (
        header
        + "\n### Ground Truth \u6bd4\u5c0d\u7d50\u679c\n"
        + f"- **Ground Truth\uff1a** {gt}\n"
        + f"- **\u6a21\u578b\u9810\u6e2c\uff1a** {predicted}\n"
        + f"- **\u6b63\u78ba\u547d\u4e2d\uff1a** {correct}\n"
        + f"- **\u6f0f\u5224\uff1a** {missed}\n"
        + f"- **\u984d\u5916\u9810\u6e2c\uff1a** {extra}\n"
        + f"- **TP / FP / FN / TN\uff1a** {prediction['tp']} / {prediction['fp']} / {prediction['fn']} / {prediction['tn']}\n"
        + f"- **Exact Match\uff1a** {'Yes' if prediction['exact_match'] else 'No'}\n"
        + f"- **Sample Precision / Recall / F1\uff1a** {prediction['sample_precision']:.4f} / "
        + f"{prediction['sample_recall']:.4f} / {prediction['sample_f1']:.4f}\n"
        + f"- **\u7d50\u679c\u89e3\u8b80\uff1a** {interpretation}"
    )
def probability_table(prediction: dict[str, Any]) -> str:
    """Render a stable, styled HTML table without depending on Gradio Dataframe DOM."""
    header_html = "".join(f"<th>{html_text(column)}</th>" for column in PROBABILITY_TABLE_COLUMNS)
    body_rows: list[str] = []
    predicted_ids_available = "predicted_class_ids" in prediction
    predicted_ids = {int(value) for value in (prediction.get("predicted_class_ids") or [])}

    for row in prediction["probability_rows"]:
        class_id = int(row["class_id"])
        probability = float(row["probability"])
        threshold = float(row["threshold"])
        is_positive = class_id in predicted_ids if predicted_ids_available else probability >= threshold
        row_class = "prob-positive" if is_positive else "prob-negative"

        # Use the backend's predicted class IDs for color, so styling does not depend
        # on whether decision text is 陽性/陰性/Positive/Negative.
        decision_text = str(row.get("decision") or ("陽性" if is_positive else "陰性"))
        cells = (
            row["class_id"],
            row["class_name_en"],
            row["class_name_zh"],
            f"{probability:.6f}",
            f"{threshold:.6f}",
            decision_text,
        )
        cells_html = "".join(f"<td>{html_text(value)}</td>" for value in cells)
        body_rows.append(f"<tr class='{row_class}'>{cells_html}</tr>")

    return (
        "<div class='probability-table-frame'>"
        "<div class='probability-table-wrap'>"
        "<table class='probability-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )


def empty_probability_table() -> str:
    return "<div class='probability-table-empty'>尚未進行分析</div>"


def log_ui_exception(output_dir: Path, stage: str, exc: BaseException) -> None:
    """Append an internal traceback without exposing it in the browser UI."""
    traceback_text = traceback.format_exc()
    if traceback_text.strip() == "NoneType: None":
        traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    entry = (
        f"[{utc_now()}] stage={stage} exception={type(exc).__name__}: {exc}\n"
        f"{traceback_text.rstrip()}\n\n"
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with _UI_ERROR_LOCK:
            with (output_dir / "ui_error.log").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(entry)
    except Exception as log_exc:
        print(f"UI traceback logging failed: {log_exc}\n{entry}", file=sys.stderr)


def analysis_ui_result(
    image_metadata: Any,
    prediction_summary: Any,
    probabilities: Any,
    probability_plot: Any,
    ground_truth: Any,
    ollama_report: Any,
    ollama_status: Any,
    system_output: Any,
    classification_seconds: Any,
    ollama_seconds: Any,
    total_seconds: Any,
    session_path: Any,
    state: Any,
    status: Any,
) -> tuple[Any, ...]:
    result = (
        image_metadata,
        prediction_summary,
        probabilities,
        probability_plot,
        ground_truth,
        ollama_report,
        ollama_status,
        system_output,
        classification_seconds,
        ollama_seconds,
        total_seconds,
        session_path,
        state,
        status,
    )
    if len(result) != ANALYSIS_OUTPUT_COUNT:
        raise RuntimeError(f"Internal UI contract error: expected {ANALYSIS_OUTPUT_COUNT} outputs")
    return result


def analysis_progress_result(status: str) -> tuple[Any, ...]:
    return analysis_ui_result(*([gr.skip()] * 13), status)


def blank_analysis_result(status: str = "等待分析") -> tuple[Any, ...]:
    return analysis_ui_result(
        None,
        "",
        empty_probability_table(),
        None,
        "",
        "",
        None,
        None,
        0.0,
        0.0,
        0.0,
        "",
        None,
        status,
    )


def error_analysis_result(message: str) -> tuple[Any, ...]:
    return analysis_ui_result(
        None,
        f"### 無法完成分析\n{message}",
        empty_probability_table(),
        None,
        "",
        "",
        {"status": "ERROR", "message": message},
        "<div class='system-grid'><div class='system-item'><span class='system-label'>系統狀態：</span><span>分析失敗</span></div></div>",
        0.0,
        0.0,
        0.0,
        "",
        None,
        f"分析失敗：{message}",
    )


def regenerate_ui_result(
    report: Any,
    audit: Any,
    ollama_seconds: Any,
    total_seconds: Any,
    status: Any,
    state: Any,
) -> tuple[Any, ...]:
    result = (report, audit, ollama_seconds, total_seconds, status, state)
    if len(result) != REGENERATE_OUTPUT_COUNT:
        raise RuntimeError(f"Internal UI contract error: expected {REGENERATE_OUTPUT_COUNT} outputs")
    return result


def image_metadata_ui(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "檔名": prediction["image_filename"],
        "原始寬度": prediction["original_width"],
        "原始高度": prediction["original_height"],
        "原始模式": prediction["original_mode"],
        "SHA256": prediction["image_sha256_short"],
        "模型輸入": "224×224 RGB",
    }


def system_status(
    inference: FullImageMultilabelInferenceService,
    ollama: FullImageMultilabelOllamaService,
    prediction: dict[str, Any] | None = None,
    ollama_audit: dict[str, Any] | None = None,
    total_seconds: float | None = None,
) -> dict[str, Any]:
    health = inference.health()
    gpu_name = None
    cuda_version = None
    if inference.device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(inference.device)
        cuda_version = torch.version.cuda
    return {
        "模型已載入": health["model_loaded"],
        "裝置": health["device"],
        "GPU": gpu_name,
        "CUDA": cuda_version,
        "checkpoint SHA256": health["model_sha256"][:12],
        "threshold SHA256": health["threshold_sha256"][:12],
        "Ollama backend": ollama.base_url,
        "Ollama model": ollama.selected_model or ollama.requested_model,
        "Ollama status": None if ollama_audit is None else ollama_audit.get("status"),
        "分類秒數": None if prediction is None else prediction["classification_seconds"],
        "Ollama 秒數": None if ollama_audit is None else ollama_audit.get("generation_seconds", 0.0),
        "總處理秒數": total_seconds,
        "singleton load count": health["model_load_count"],
        "inference lock": health["inference_lock"],
        "queue concurrency": 1,
    }


def format_system_status_html(payload: dict[str, Any]) -> str:
    model_ok = bool(payload.get("模型已載入"))
    ollama_state = payload.get("Ollama status")
    ollama_ok = ollama_state in (None, "PASS")
    return (
        "<div class='system-grid'>"
        f"<div class='system-item'><span class='system-label'>Inference Service：</span><span class='system-value-ok'>{'正常' if model_ok else '異常'}</span></div>"
        f"<div class='system-item'><span class='system-label'>GPU：</span><span>{html_text(payload.get('GPU') or payload.get('裝置') or 'N/A')}</span></div>"
        f"<div class='system-item'><span class='system-label'>Ollama Service：</span><span class='system-value-ok'>{'正常' if ollama_ok else '暫時無法使用'}（{html_text(payload.get('Ollama model') or 'N/A')}）</span></div>"
        f"<div class='system-item'><span class='system-label'>CUDA：</span><span>{html_text(payload.get('CUDA') or 'N/A')}</span></div>"
        "</div>"
    )


def format_analysis_status_html(
    prediction: dict[str, Any] | None = None,
    ollama_audit: dict[str, Any] | None = None,
    message: str = "等待分析",
) -> str:
    if prediction is None:
        return f"<div class='status-complete'>{html_text(message)}</div>"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ollama_seconds = 0.0 if ollama_audit is None else float(ollama_audit.get("generation_seconds", 0.0) or 0.0)
    return (
        "<div class='status-complete'>◉ 分析完成</div>"
        "<div class='status-grid'>"
        f"<span class='status-label'>影像檔案：</span><span>{html_text(prediction.get('image_filename') or 'N/A')}</span>"
        f"<span class='status-label'>分析時間：</span><span>{generated}</span>"
        f"<span class='status-label'>推論時間：</span><span>{float(prediction.get('classification_seconds') or 0.0):.2f} sec</span>"
        f"<span class='status-label'>Ollama 回應時間：</span><span>{ollama_seconds:.2f} sec</span>"
        "</div>"
    )



def analyze_stream(
    image: Any,
    record_number: str,
    name: str,
    sex: str,
    age: Any,
    exam_date: str,
    note: str,
    inference: FullImageMultilabelInferenceService,
    ollama: FullImageMultilabelOllamaService,
    output_dir: Path,
) -> Iterator[tuple[Any, ...]]:
    yield analysis_progress_result("<div class='status-complete'>正在分析影像…</div>")
    total_started = time.perf_counter()
    try:
        prediction = inference.predict(image)
        structured = inference.ollama_payload(prediction)
        probability_figure = create_probability_figure(prediction)
    except Exception as exc:
        log_ui_exception(output_dir, "classification_or_ui_preparation", exc)
        yield error_analysis_result("模型分析失敗，請確認上傳的是可讀取的完整胸腔 X 光圖片。")
        return
    state = {"ollama_payload": structured, "session_dir": None, "prediction": prediction}
    yield analysis_ui_result(
        image_metadata_ui(prediction),
        format_prediction_summary(prediction),
        probability_table(prediction),
        probability_figure,
        format_ground_truth(prediction, inference.catalog_path),
        gr.skip(),
        gr.skip(),
        format_system_status_html(system_status(inference, ollama, prediction)),
        prediction["classification_seconds"],
        0.0,
        prediction["total_model_seconds"],
        "",
        state,
        format_analysis_status_html(prediction, None, "模型分析完成"),
    )
    yield analysis_progress_result("<div class='status-complete'>模型完成，正在生成 AI 說明…</div>")
    try:
        patient = patient_info(record_number, name, sex, age, exam_date, note)
        report, ollama_audit = generate_report_safe(ollama, structured, patient)
    except Exception as exc:
        log_ui_exception(output_dir, "ollama_wrapper", exc)
        report = OLLAMA_FALLBACK
        ollama_audit = {
            "status": "UNAVAILABLE",
            "backend": ollama.base_url,
            "model": ollama.selected_model or ollama.requested_model,
            "generation_seconds": 0.0,
            "error_type": type(exc).__name__,
            "error_message": "Ollama auxiliary explanation failed",
            "image_sent_to_ollama": False,
            "ground_truth_sent_to_ollama": False,
        }
    total_seconds = time.perf_counter() - total_started
    session_dir: Path | None = None
    session_error = False
    try:
        session_dir = persist_session(
            output_dir,
            prediction,
            structured,
            report,
            ollama_audit,
            patient,
            total_seconds,
            smoke_test=False,
        )
    except Exception as exc:
        session_error = True
        log_ui_exception(output_dir, "session_persistence", exc)
    state = {
        "ollama_payload": structured,
        "session_dir": None if session_dir is None else str(session_dir),
        "prediction": prediction,
    }
    system_payload = system_status(inference, ollama, prediction, ollama_audit, total_seconds)
    system_payload["session_write_status"] = "FAILED" if session_error else "PASS"
    if session_error:
        final_status = "分析完成，但 session 儲存失敗"
    elif ollama_audit.get("status") == "PASS":
        final_status = "全部完成"
    else:
        final_status = "模型完成，Ollama 暫時無法使用"
    yield analysis_ui_result(
        image_metadata_ui(prediction),
        format_prediction_summary(prediction),
        probability_table(prediction),
        probability_figure,
        format_ground_truth(prediction, inference.catalog_path),
        report_sheet_html(report, prediction, patient, session_dir),
        {key: value for key, value in ollama_audit.items() if key != "response"},
        format_system_status_html(system_payload),
        prediction["classification_seconds"],
        ollama_audit.get("generation_seconds", 0.0),
        total_seconds,
        "" if session_dir is None else str(session_dir),
        state,
        format_analysis_status_html(prediction, ollama_audit, final_status),
    )


def regenerate_report(
    state: dict[str, Any] | None,
    record_number: str,
    name: str,
    sex: str,
    age: Any,
    exam_date: str,
    note: str,
    ollama: FullImageMultilabelOllamaService,
    output_dir: Path,
) -> tuple[Any, ...]:
    if not state or not state.get("ollama_payload"):
        return regenerate_ui_result(
            "請先完成模型推論，再重新產生 Ollama 說明。",
            {"status": "NOT_READY"},
            0.0,
            0.0,
            "尚未完成模型推論",
            state,
        )
    started = time.perf_counter()
    try:
        patient = patient_info(record_number, name, sex, age, exam_date, note)
        report, audit = generate_report_safe(ollama, state["ollama_payload"], patient)
        session_dir_value = state.get("session_dir")
        session_write_failed = False
        if session_dir_value:
            try:
                session_dir = Path(session_dir_value)
                suffix = timestamp_slug()
                atomic_write_text(
                    session_dir / f"ollama_report_regenerated_{suffix}.txt",
                    report.rstrip() + "\n",
                )
                atomic_write_json(
                    session_dir / f"ollama_audit_regenerated_{suffix}.json",
                    {key: value for key, value in audit.items() if key != "response"},
                )
                write_report_artifacts(session_dir, state.get("prediction"), patient, report)
            except Exception as exc:
                session_write_failed = True
                log_ui_exception(output_dir, "regenerate_session_persistence", exc)
    except Exception as exc:
        log_ui_exception(output_dir, "regenerate_report", exc)
        elapsed = time.perf_counter() - started
        return regenerate_ui_result(
            blank_report_sheet_html(OLLAMA_FALLBACK),
            {"status": "UNAVAILABLE", "error_type": type(exc).__name__},
            0.0,
            elapsed,
            "Ollama 暫時無法使用",
            state,
        )
    elapsed = time.perf_counter() - started
    if session_write_failed:
        status = "Ollama 說明已產生，但 session 儲存失敗"
    elif audit.get("status") == "PASS":
        status = "Ollama 說明已重新產生"
    else:
        status = "Ollama 暫時無法使用"
    return regenerate_ui_result(
        report_sheet_html(report, state.get("prediction") if state else None, patient, Path(state["session_dir"]) if state and state.get("session_dir") else None),
        {key: value for key, value in audit.items() if key != "response"},
        audit.get("generation_seconds", 0.0),
        elapsed,
        status,
        state,
    )


def fixed_examples(output_catalog_dir: Path) -> list[list[str]]:
    demo_dir = output_catalog_dir / "demo_images"
    return [[str(path)] for path in sorted(demo_dir.glob("*.png"))[:6]] if demo_dir.is_dir() else []


def build_demo(
    inference: FullImageMultilabelInferenceService,
    ollama: FullImageMultilabelOllamaService,
    output_dir: Path,
    catalog_dir: Path,
) -> gr.Blocks:
    try:
        top_ollama_health = ollama.health()
        top_ollama_ok = top_ollama_health.get("status") == "PASS"
    except Exception:
        top_ollama_ok = False
    # Top bar intentionally stays free of backend jargon (Ollama, model codename,
    # SHA256, etc.) — that detail lives in the "系統狀態" card at the bottom right,
    # which is the appropriate place for maintainers, not the page clinicians see first.
    topbar = (
        "<header class='app-topbar'>"
        "<div><div class='app-title-row'>"
        "<h1>AI 輔助胸腔 X 光多標籤辨識系統</h1>"
        "<span class='research-pill'>研究用途</span>"
        "</div>"
        "<p>五類胸腔 X 光異常輔助偵測與 AI 說明產生</p>"
        "</div>"
        "<div class='top-health'>"
        f"<div class='health-ok'><span class='health-dot'>●</span>AI 輔助說明服務：{'正常運作中' if top_ollama_ok else '暫時無法使用'}</div>"
        "</div></header>"
    )

    with gr.Blocks(title="AI 輔助胸腔 X 光多標籤辨識系統") as demo:
        with gr.Column(elem_classes=["app-shell"], elem_id="app-root"):
            gr.HTML(topbar, elem_classes=["topbar-host"])

            with gr.Column(elem_classes=["ui-card", "patient-card"]):
                gr.HTML(
                    "<div class='ui-card-header'><h3>"
                    "<svg class='header-icon' width='16' height='16' viewBox='0 0 24 24' fill='none' "
                    "stroke='currentColor' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
                    "<circle cx='12' cy='8' r='3.6'></circle>"
                    "<path d='M5 20c0-3.6 3.1-6.2 7-6.2s7 2.6 7 6.2'></path>"
                    "</svg>病人資訊</h3></div>",
                    elem_classes=["card-header-host"],
                )
                with gr.Row(elem_classes=["ui-card-body", "patient-fields"]):
                    record_number = gr.Textbox(label="病歷號碼", max_lines=1, scale=1)
                    patient_name = gr.Textbox(label="姓名", max_lines=1, scale=1)
                    patient_sex = gr.Dropdown(choices=["男性", "女性"], value=None, label="性別", scale=1)
                    patient_age = gr.Number(label="年齡", precision=0, minimum=0, maximum=130, scale=1)
                    exam_date = gr.Textbox(label="檢查日期", max_lines=1, scale=1)
                    patient_note = gr.Textbox(label="備註", max_lines=1, scale=2, elem_classes=["patient-note-field"])

            with gr.Row(equal_height=False, elem_classes=["main-grid"]):
                with gr.Column(scale=1, elem_classes=["left-pane"]):
                    with gr.Column(elem_classes=["image-card"]):
                        gr.HTML(
                            "<div class='ui-card-header'>"
                            "<h3>完整胸腔 X 光影像</h3>"
                            "</div>",
                            elem_classes=["card-header-host"],
                        )

                        with gr.Column(elem_classes=["image-host"]):
                            image_input = gr.Image(
                                type="filepath",
                                image_mode=None,
                                sources=["upload"],
                                label=None,
                                show_label=False,
                                height=390,
                                elem_id="cxr-image-input",
                            )

                            # 保留為隱藏輸出，避免影響原本 callback 輸出順序。
                            image_metadata = gr.JSON(
                                value=None,
                                label=None,
                                visible=False,
                            )

                        with gr.Row(elem_classes=["action-row"]):
                            analyze_button = gr.Button(
                                "▶ 開始分析",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )
                            regenerate_button = gr.Button(
                                "↻ 重新產生說明",
                                elem_classes=["secondary-action"],
                            )
                            clear_button = gr.Button(
                                "✕ 清除",
                                elem_classes=["danger-action"],
                            )

                    with gr.Column(elem_classes=["ui-card"]):
                        gr.HTML(
                            "<div class='ui-card-header'><h3>分析狀態</h3></div>",
                            elem_classes=["card-header-host"],
                        )
                        with gr.Column(elem_classes=["ui-card-body"]):
                            status = gr.HTML(
                                value=format_analysis_status_html(message="等待分析"),
                                elem_classes=["transparent-output", "analysis-status"],
                            )

                    with gr.Column(elem_classes=["ui-card"]):
                        gr.HTML(
                            "<div class='ui-card-header'>"
                            "<h3>五類疾病機率（Validation Threshold）</h3>"
                            "</div>",
                            elem_classes=["card-header-host"],
                        )
                        with gr.Column(elem_classes=["ui-card-body", "probability-card-body"]):
                            probability_dataframe = gr.HTML(
                                value=empty_probability_table(),
                                elem_id="probability-table-host",
                                elem_classes=["probability-table-host"],
                            )
                            with gr.Accordion(
                                "查看機率圖",
                                open=False,
                                elem_classes=["app-accordion"],
                            ):
                                probability_plot = gr.Plot(
                                    label=None,
                                    show_label=False,
                                )

                    with gr.Accordion(
                        "Ground Truth 比對（僅供研發驗證用，一般看診不需要展開）",
                        open=False,
                        elem_classes=["ui-card", "gt-accordion", "app-accordion"],
                    ):
                        with gr.Column(elem_classes=["ui-card-body"]):
                            ground_truth_output = gr.Markdown(
                                elem_classes=["transparent-output", "gt-compact"]
                            )

                with gr.Column(scale=1, elem_classes=["right-pane"]):
                    # 保留為隱藏輸出，維持既有分析 callback 的輸出順序。
                    prediction_summary = gr.Markdown(
                        value="",
                        visible=False,
                    )

                    with gr.Column(
                        elem_classes=["ui-card", "report-card"],
                        elem_id="report-card",
                    ):
                        with gr.Row(
                            elem_classes=["ui-card-header", "report-toolbar"]
                        ):
                            gr.HTML(
                                "<div class='header-copy'>"
                                "<h3>AI 輔助胸腔 X 光診斷說明書</h3>"
                                "</div>",
                                elem_classes=["card-header-host"],
                            )
                            print_button = gr.Button(
                                "▣ 列印／匯出 PDF",
                                elem_classes=["sheet-print-button", "no-print"],
                            )
                        with gr.Column(elem_classes=["report-body"]):
                            ollama_report = gr.HTML(
                                value=blank_report_sheet_html(),
                                elem_classes=["report-html-host"],
                            )

                    # 保留為隱藏輸出，維持分析與重新產生 callback 的既有輸出順序。
                    system_output = gr.HTML(
                        value=format_system_status_html(system_status(inference, ollama)),
                        visible=False,
                    )
                    ollama_status = gr.JSON(value=None, visible=False)
                    classification_seconds = gr.Number(value=0.0, visible=False)
                    ollama_seconds = gr.Number(value=0.0, visible=False)
                    total_seconds = gr.Number(value=0.0, visible=False)
                    session_path = gr.Textbox(value="", visible=False)

            last_result_state = gr.State(value=None)

            analysis_outputs = [
                image_metadata,
                prediction_summary,
                probability_dataframe,
                probability_plot,
                ground_truth_output,
                ollama_report,
                ollama_status,
                system_output,
                classification_seconds,
                ollama_seconds,
                total_seconds,
                session_path,
                last_result_state,
                status,
            ]

            def analyze_event(
                image: Any,
                record: str,
                name: str,
                sex: str,
                age: Any,
                date: str,
                note: str,
            ) -> Iterator[tuple[Any, ...]]:
                yield from analyze_stream(
                    image, record, name, sex, age, date, note,
                    inference, ollama, output_dir,
                )

            def regenerate_event(
                state: dict[str, Any] | None,
                record: str,
                name: str,
                sex: str,
                age: Any,
                date: str,
                note: str,
            ) -> tuple[Any, ...]:
                return regenerate_report(
                    state, record, name, sex, age, date, note,
                    ollama, output_dir,
                )

            if not inspect.isgeneratorfunction(analyze_event):
                raise RuntimeError("Analysis callback must remain a real generator function")

            queued = analyze_button.click(
                fn=lambda: "<div class='status-complete'>已加入排隊</div>",
                inputs=None,
                outputs=status,
                queue=False,
                api_visibility="private",
            )
            analyzed = queued.then(
                fn=analyze_event,
                inputs=[image_input, record_number, patient_name, patient_sex, patient_age, exam_date, patient_note],
                outputs=analysis_outputs,
                show_progress="full",
                concurrency_limit=1,
                concurrency_id="full_image_analysis",
                api_visibility="private",
            )

            print_button.click(
                fn=None,
                inputs=None,
                outputs=None,
                js="() => { window.print(); return []; }",
                queue=False,
                api_visibility="private",
            )

            regenerate_button.click(
                fn=regenerate_event,
                inputs=[last_result_state, record_number, patient_name, patient_sex, patient_age, exam_date, patient_note],
                outputs=[ollama_report, ollama_status, ollama_seconds, total_seconds, status, last_result_state],
                show_progress="full",
                concurrency_limit=1,
                concurrency_id="full_image_analysis",
                api_visibility="private",
            )

            def clear_all() -> tuple[Any, ...]:
                return (
                    None,
                    "",
                    "",
                    None,
                    None,
                    "",
                    "",
                    "",
                    empty_probability_table(),
                    None,
                    "",
                    blank_report_sheet_html(),
                    None,
                    format_system_status_html(system_status(inference, ollama)),
                    0.0,
                    0.0,
                    0.0,
                    "",
                    None,
                    format_analysis_status_html(message="等待分析"),
                )

            clear_button.click(
                fn=clear_all,
                inputs=None,
                outputs=[
                    image_input,
                    record_number,
                    patient_name,
                    patient_sex,
                    patient_age,
                    exam_date,
                    patient_note,
                    prediction_summary,
                    probability_dataframe,
                    probability_plot,
                    ground_truth_output,
                    ollama_report,
                    ollama_status,
                    system_output,
                    classification_seconds,
                    ollama_seconds,
                    total_seconds,
                    session_path,
                    last_result_state,
                    status,
                ],
                queue=False,
                api_visibility="private",
            )

    demo.queue(default_concurrency_limit=1, max_size=20)
    return demo


def first_readable_validation(project_root: Path) -> tuple[Path, list[int], dict[str, str]]:
    manifest = (project_root / DEFAULT_VAL_MANIFEST).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Validation manifest is missing: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = Path(row["full_image_path"]).resolve()
        try:
            with Image.open(path) as image:
                image.load()
        except Exception:
            continue
        truth = [int(row[field]) for field in (
            "label_0_aortic_enlargement",
            "label_1_cardiomegaly",
            "label_2_pleural_thickening",
            "label_3_pulmonary_fibrosis",
            "label_4_pleural_effusion",
        )]
        return path, truth, row
    raise RuntimeError("No readable Validation full image was found")


class _UnavailableOllama:
    base_url = "http://127.0.0.1:9"
    requested_model = "gemma3:4b"
    selected_model = None

    def generate(self, structured_result: dict[str, Any], patient: dict[str, Any]) -> dict[str, Any]:
        del structured_result, patient
        raise FullImageOllamaError("simulated unavailable backend")


def run_smoke_test(
    args: argparse.Namespace,
    inference: FullImageMultilabelInferenceService,
    ollama: FullImageMultilabelOllamaService,
) -> dict[str, Any]:
    image_path, expected_truth, row = first_readable_validation(args.project_root)
    source_sha_before = sha256_file(image_path)
    started = time.perf_counter()
    prediction = inference.predict(image_path)
    if prediction["ground_truth_label_vector"] != expected_truth:
        raise RuntimeError(
            f"Ground Truth catalog mismatch: {prediction['ground_truth_label_vector']} != {expected_truth}"
        )
    structured = inference.ollama_payload(prediction)
    messages, prompt_hash = build_report_messages(
        patient_info=patient_info("", "", "", None, "", ""), **structured
    )
    prompt_text = json.dumps(messages, ensure_ascii=False)
    if "Ground Truth" in prompt_text or any("ground_truth" in key for key in structured):
        raise RuntimeError("Ground Truth leaked into the Ollama prompt")
    report, ollama_audit = generate_report_safe(
        ollama, structured, patient_info("", "", "", None, "", "")
    )
    ollama_available_health = ollama.health()["status"]
    ollama_available_generation_retained = True
    fallback_report, fallback_audit = generate_report_safe(
        _UnavailableOllama(), structured, patient_info("", "", "", None, "", "")
    )
    if not fallback_report.startswith(OLLAMA_FALLBACK) or fallback_audit.get("status") != "UNAVAILABLE":
        raise RuntimeError("Ollama unavailable fallback did not preserve classification")
    total_seconds = time.perf_counter() - started
    session_dir = persist_session(
        args.output_dir,
        prediction,
        structured,
        report,
        ollama_audit,
        patient_info("", "", "", None, "", ""),
        total_seconds,
        smoke_test=True,
    )
    required = {
        "input_metadata.json",
        "model_prediction.json",
        "model_prediction.csv",
        "ground_truth_comparison.json",
        "ollama_report.txt",
        "ollama_audit.json",
        "session_audit.json",
    }
    actual = {path.name for path in session_dir.iterdir() if path.is_file()}
    if not required.issubset(actual):
        raise RuntimeError(f"Smoke session files missing: {sorted(required - actual)}")
    if (session_dir / "model_prediction.csv").read_bytes()[:3] != b"\xef\xbb\xbf":
        raise RuntimeError("Smoke session CSV is not UTF-8 BOM")
    if sha256_file(image_path) != source_sha_before:
        raise RuntimeError("Validation source image changed during smoke test")
    remnants = [
        str(path)
        for path in args.output_dir.rglob("*")
        if path.name.endswith((".tmp", ".writing"))
    ]
    if remnants:
        raise RuntimeError(f"Temporary output remnants found: {remnants}")
    smoke = {
        "status": "PASS",
        "timestamp": utc_now(),
        "image_path": str(image_path),
        "image_id": row["image_id"],
        "input_tensor_shape": prediction["input_tensor_shape"],
        "logits_shape": prediction["logits_shape"],
        "probabilities_shape": prediction["probabilities_shape"],
        "predicted_vector_shape": prediction["predicted_vector_shape"],
        "probabilities": prediction["class_probabilities"],
        "thresholds": prediction["class_thresholds"],
        "predicted_label_vector": prediction["predicted_label_vector"],
        "predicted_class_ids": prediction["predicted_class_ids"],
        "predicted_class_names_en": prediction["predicted_class_names_en"],
        "ground_truth_label_vector": prediction["ground_truth_label_vector"],
        "ground_truth_class_ids": prediction["ground_truth_class_ids"],
        "tp": prediction["tp"],
        "fp": prediction["fp"],
        "fn": prediction["fn"],
        "tn": prediction["tn"],
        "exact_match": prediction["exact_match"],
        "sample_precision": prediction["sample_precision"],
        "sample_recall": prediction["sample_recall"],
        "sample_f1": prediction["sample_f1"],
        "prompt_sha256": prompt_hash,
        "ground_truth_sent_to_ollama": False,
        "image_sent_to_ollama": False,
        "ollama_available_test": ollama_audit.get("status"),
        "ollama_available_health": ollama_available_health,
        "ollama_available_generation_retained": ollama_available_generation_retained,
        "ollama_model": ollama_audit.get("model"),
        "ollama_generation_seconds": ollama_audit.get("generation_seconds", 0.0),
        "ollama_unavailable_test": "PASS",
        "classification_retained_when_ollama_unavailable": True,
        "classification_seconds": prediction["classification_seconds"],
        "total_seconds": total_seconds,
        "session_dir": display_path(session_dir, args.project_root),
        "test_images_read_count": 0,
        "uses_bbox": False,
        "uses_roi_crop": False,
        "source_image_unchanged": True,
        "temporary_remnants": [],
        "offline_smoke_test": bool(getattr(args, "offline_smoke_test", False)),
        "external_network_required": False,
        "cloud_api_called": False,
        "gradio_share": False,
        "gradio_live_url_created": False,
        "server_name": args.server_name,
        "requested_port": args.server_port,
        "port_failover_test": verify_port_failover(args.server_name),
    }
    smoke = sanitize_for_audit(smoke, args.project_root)
    atomic_write_json(args.output_dir / "smoke_test_audit.json", smoke)
    return smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_RELATIVE)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_RELATIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_RELATIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_RELATIVE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="gemma3:4b")
    parser.add_argument("--ollama-timeout", type=float, default=120.0)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--offline-smoke-test", action="store_true")
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Bind to 0.0.0.0 for trusted LAN demos. share remains False.",
    )
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.project_root = args.project_root.expanduser().resolve()
    if not args.project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {args.project_root}")
    for name in ("model", "thresholds", "catalog", "output_dir"):
        setattr(args, name, resolve_path(args.project_root, getattr(args, name)))
    if args.lan:
        args.server_name = "0.0.0.0"
    elif args.server_name != "127.0.0.1":
        raise ValueError("server-name must remain 127.0.0.1 unless --lan is used")
    active_modes = sum(bool(value) for value in (args.dry_run, args.smoke_test, args.offline_smoke_test))
    if active_modes > 1:
        raise ValueError("Choose only one of --dry-run, --smoke-test, or --offline-smoke-test")
    return args


def startup_audit(
    args: argparse.Namespace,
    inference: FullImageMultilabelInferenceService,
    ollama: FullImageMultilabelOllamaService,
    selected_port: int,
    port_state: dict[str, Any],
) -> dict[str, Any]:
    queue_signature = str(inspect.signature(gr.Blocks.queue))
    click_signature = str(inspect.signature(gr.Button.click))
    launch_signature = str(inspect.signature(gr.Blocks.launch))
    if "default_concurrency_limit" not in queue_signature or "max_size" not in queue_signature:
        raise RuntimeError("Installed Gradio queue API lacks required parameters")
    if "concurrency_limit" not in click_signature or "concurrency_id" not in click_signature:
        raise RuntimeError("Installed Gradio event API lacks concurrency controls")
    health = inference.health()
    model_sha_match = health["model_sha256"] == EXPECTED_FORMAL_MODEL_SHA256
    threshold_sha_match = health["threshold_sha256"] == EXPECTED_THRESHOLD_SHA256
    singleton_again = get_inference_service(
        args.model, args.thresholds, args.device, args.catalog
    )
    if singleton_again is not inference or inference.model_load_count != 1:
        raise RuntimeError("Inference singleton loading validation failed")
    ollama_health = ollama.health()
    audit = {
        "status": "PASS",
        "timestamp": utc_now(),
        "dry_run": args.dry_run,
        "smoke_test": args.smoke_test,
        "offline_smoke_test": args.offline_smoke_test,
        "offline_local_only": not args.lan,
        "lan_mode": args.lan,
        "gradio_version": gr.__version__,
        "gradio_queue_signature": queue_signature,
        "gradio_event_signature": click_signature,
        "gradio_launch_signature": launch_signature,
        "queue": {
            "default_concurrency_limit": 1,
            "max_size": 20,
            "analysis_concurrency_limit": 1,
            "analysis_concurrency_id": "full_image_analysis",
        },
        "checkpoint": {
            "exists": args.model.is_file(),
            "path": args.model,
            "sha256": health["model_sha256"],
            "expected_sha256": EXPECTED_FORMAL_MODEL_SHA256,
            "sha256_match": model_sha_match,
        },
        "threshold": {
            "exists": args.thresholds.is_file(),
            "path": args.thresholds,
            "sha256": health["threshold_sha256"],
            "expected_sha256": EXPECTED_THRESHOLD_SHA256,
            "sha256_match": threshold_sha_match,
        },
        "model": health,
        "ollama": ollama_health,
        "ground_truth_catalog": {
            "available": health["catalog_available"],
            "path": health["catalog_path"],
            "rows": health["catalog_rows"],
        },
        "output_write_permission": "PASS",
        "output_dir": args.output_dir,
        "server": {
            "server_name": args.server_name,
            "requested_port": args.server_port,
            "selected_port": selected_port,
            **port_state,
            "ports": port_statuses(args.server_name),
            "urls": launch_urls(args.server_name, selected_port, args.lan),
            "share": False,
            "show_error": False,
            "depends_on_gradio_live": False,
            "cloud_api_dependency": False,
        },
        "dependencies": dependency_audit(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            **cuda_audit(inference.device),
        },
        "prompt_schema_sha256": prompt_schema_sha256(),
        "test_images_read_count": 0,
        "model_inference_count": 0,
        "ollama_generation_count": 0,
        "optimizer_created": False,
        "backward_executed": False,
        "flask_application_context_used": False,
    }
    if not model_sha_match or not threshold_sha_match:
        raise RuntimeError("Formal checkpoint or threshold SHA256 verification failed")
    return sanitize_for_audit(audit, args.project_root)


def main() -> int:
    validate_clinical_reference_rules()
    args = resolve_args(parse_args())
    if not args.model.is_file() or not args.thresholds.is_file():
        raise RuntimeError("Formal Full-image model or Validation thresholds are missing")
    check_output_write_permission(args.output_dir)
    inference = get_inference_service(args.model, args.thresholds, args.device, args.catalog)
    ollama = FullImageMultilabelOllamaService(
        args.ollama_base_url, args.ollama_model, args.ollama_timeout
    )
    selected_port, port_state = select_port(args.server_name, args.server_port)
    audit = startup_audit(args, inference, ollama, selected_port, port_state)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if args.smoke_test or args.offline_smoke_test:
        smoke = run_smoke_test(args, inference, ollama)
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return 0

    atomic_write_json(args.output_dir / "app_startup_audit.json", audit)
    demo = build_demo(
        inference,
        ollama,
        args.output_dir,
        args.catalog.parent,
    )
    urls = launch_urls(args.server_name, selected_port, args.lan)
    print("=" * 72)
    print(f"UI layout version: {UI_LAYOUT_VERSION}")
    print(f"Full-image multilabel Demo local URL: {urls['local_url']}")
    if args.lan:
        print(f"LAN URL: {urls['lan_url']}")
        print("LAN mode warning: use only on a trusted local network; share remains False.")
    print("Offline mode: share=False; no gradio.live URL; no cloud LLM API.")
    print("=" * 72)
    demo.launch(
        server_name=args.server_name,
        server_port=selected_port,
        share=False,
        show_error=False,
        inbrowser=args.inbrowser,
        footer_links=[],
        allowed_paths=[str(args.catalog.parent / "demo_images"), str(args.output_dir)],
        max_file_size="30mb",
        enable_monitoring=False,
        strict_cors=True,
        mcp_server=False,
        quiet=False,
        css=APP_CSS,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
