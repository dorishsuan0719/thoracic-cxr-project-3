"""Single source of truth for Full-image multilabel Ollama report prompts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CLASS_MAPPING_EN = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
CLASS_MAPPING_ZH = {
    0: "主動脈擴大",
    1: "心臟擴大",
    2: "胸膜增厚",
    3: "肺纖維化",
    4: "胸腔積液",
}
DISCLAIMER = "本結果僅供研究與教學展示，不可取代醫師判讀或臨床診斷。"
NO_POSITIVE_MESSAGE = "五個目標類別中，沒有任何類別達到其 Validation 判定門檻。"
REQUIRED_HEADINGS = (
    "## 模型辨識摘要",
    "## 達門檻類別說明",
    "## 未達門檻類別摘要",
    "## 模型限制",
    "## 建議",
    "## 研究用途警語",
)


def _clean(value: Any, maximum: int = 200) -> str:
    text = " ".join(str(value or "").split())
    return text[:maximum] if text else "未提供"


def build_report_messages(
    *,
    patient_info: dict[str, Any],
    class_probabilities: list[float],
    class_thresholds: list[float],
    predicted_class_ids: list[int],
    predicted_class_names_en: list[str],
    predicted_class_names_zh: list[str],
    preprocessing_description: str,
    model_name: str,
    disclaimer: str,
) -> tuple[list[dict[str, str]], str]:
    if len(class_probabilities) != 5 or len(class_thresholds) != 5:
        raise ValueError("Exactly five probabilities and thresholds are required")
    if any(not 0.0 <= float(value) <= 1.0 for value in class_probabilities + class_thresholds):
        raise ValueError("Probabilities and thresholds must be within [0,1]")
    if sorted(set(predicted_class_ids)) != predicted_class_ids:
        raise ValueError("predicted_class_ids must be unique and sorted")
    if any(class_id not in CLASS_MAPPING_EN for class_id in predicted_class_ids):
        raise ValueError("predicted_class_ids must be within 0..4")
    expected_en = [CLASS_MAPPING_EN[class_id] for class_id in predicted_class_ids]
    expected_zh = [CLASS_MAPPING_ZH[class_id] for class_id in predicted_class_ids]
    if predicted_class_names_en != expected_en or predicted_class_names_zh != expected_zh:
        raise ValueError("Predicted class names do not match the fixed class mapping")
    if disclaimer != DISCLAIMER:
        raise ValueError("The required research disclaimer was changed")

    patient_lines = [
        f"病歷號碼：{_clean(patient_info.get('record_number'), 80)}",
        f"姓名：{_clean(patient_info.get('name'), 80)}",
        f"性別：{_clean(patient_info.get('sex'), 30)}",
        f"年齡：{_clean(patient_info.get('age'), 20)}",
        f"檢查日期：{_clean(patient_info.get('exam_date'), 40)}",
        f"備註：{_clean(patient_info.get('note'), 300)}",
    ]
    class_lines = []
    predicted_set = set(predicted_class_ids)
    for class_id in range(5):
        decision = "Positive，達到門檻" if class_id in predicted_set else "Negative，未達門檻"
        class_lines.append(
            f"Class {class_id} | {CLASS_MAPPING_EN[class_id]} | {CLASS_MAPPING_ZH[class_id]} | "
            f"probability={float(class_probabilities[class_id]):.6f} | "
            f"Validation threshold={float(class_thresholds[class_id]):.6f} | {decision}"
        )
    positive_text = (
        "、".join(
            f"{zh}（{en}）"
            for en, zh in zip(predicted_class_names_en, predicted_class_names_zh)
        )
        if predicted_class_ids
        else NO_POSITIVE_MESSAGE
    )

    system_message = f"""你是研究展示系統的繁體中文文字整理助手。你不看圖片、不做影像判讀，也不能修改分類模型輸出。

必須遵守：
1. 僅依照提供的五類 probability、Validation threshold 與 Positive/Negative 結果撰寫。
2. 不得新增模型未預測為 Positive 的疾病，不得修改任何數值或門檻。
3. 不得宣稱確診、正常、健康、未發現異常，不得提供治療或用藥指示。
4. 病人資訊只用於報告脈絡，不可影響模型判定，也不要重複可識別身分資訊。
5. 不得提及或推測資料集答案、Test 答案、BBox、ROI 或隱藏標籤。
6. 必須使用繁體中文，並依序保留下列六個 Markdown 標題：
{chr(10).join(REQUIRED_HEADINGS)}
7. 最後一節必須逐字包含：{DISCLAIMER}
"""
    user_message = f"""請根據以下固定結構化結果產生簡潔的研究用輔助說明。

模型名稱：{_clean(model_name, 160)}
前處理：{_clean(preprocessing_description, 300)}

選填病人資訊（不得影響分類，也不要在回覆中重複姓名或病歷號碼）：
{chr(10).join(patient_lines)}

五類獨立 Sigmoid 結果：
{chr(10).join(class_lines)}

達門檻類別：{positive_text}

撰寫要求：
- 「模型辨識摘要」只整理模型輸出，不做診斷。
- 「達門檻類別說明」只說明上列 Positive 類別；若沒有 Positive，逐字使用指定的沒有達門檻訊息。
- 「未達門檻類別摘要」可列出 Negative 類別，但不得稱為正常或排除疾病。
- 「模型限制」說明 full-image 224×224 resize、多標籤模型與資料限制，且模型不輸出病灶框。
- 「建議」只能建議由醫師結合原始影像與臨床資訊進一步判讀。
- 不要輸出資料集標註、TP/FP/FN/TN 或任何資料集答案。
"""
    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]
    canonical = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return messages, hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def prompt_schema_sha256() -> str:
    payload = {
        "classes_en": CLASS_MAPPING_EN,
        "classes_zh": CLASS_MAPPING_ZH,
        "required_headings": REQUIRED_HEADINGS,
        "disclaimer": DISCLAIMER,
        "no_positive_message": NO_POSITIVE_MESSAGE,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
