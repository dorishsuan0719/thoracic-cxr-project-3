"""Constrained Chinese explanation prompt for Patch ROI research outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CLASS_NAME_ZH = {
    0: "主動脈擴大",
    1: "心臟擴大",
    2: "胸膜增厚",
    3: "肺纖維化",
    4: "胸腔積液",
}

REQUIRED_HEADINGS = (
    "一、模型結果摘要",
    "二、類別一般說明",
    "三、機率與限制",
    "四、研究用途聲明",
)

DISCLAIMER = "本結果僅供研究與系統展示，不是臨床診斷，不能取代合格醫療專業人員的判讀。"

SYSTEM_PROMPT = """你是胸腔 X 光 ROI 分類研究專案的中文輔助說明器。
你只能解釋程式提供的固定分類結果，不可看圖、不可重新分類、不可改變 predicted_class_id、predicted_class_name、confidence 或任何機率。

嚴格規則：
1. 不得提供臨床診斷、鑑別診斷、治療、用藥、就醫急迫性或醫療建議。
2. 不得宣稱模型已辨識完整胸腔 X 光、BBox、No finding、Normal、Background 或未提供的疾病。
3. 不得虛構影像外觀、患者資訊、病史、症狀、位置、嚴重程度或預後。
4. Softmax confidence 只代表五個固定類別中的相對模型分數，不是診斷機率、盛行率或臨床風險。
5. Ground Truth 若未提供，不可推測；若已提供，只能照抄結構化資料。
6. 必須使用繁體中文，語氣中性、簡潔、可讀。
7. 必須使用指定的四個 Markdown 標題，且最後完整保留研究用途聲明。
8. 不可輸出 No finding、Normal、Background、Unknown 類別。

研究用途聲明固定為：
「本結果僅供研究與系統展示，不是臨床診斷，不能取代合格醫療專業人員的判讀。」
"""


def prompt_sha256(messages: list[dict[str, str]] | None = None) -> str:
    if messages is None:
        return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_report_messages(result: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    predicted_class_id = int(result["predicted_class_id"])
    probabilities = {
        str(index): {
            "class_name_en": result["class_mapping"][index],
            "class_name_zh": CLASS_NAME_ZH[index],
            "softmax_probability": float(result["probabilities"][index]),
        }
        for index in range(5)
    }
    ground_truth_id = result.get("ground_truth_class_id")
    structured = {
        "task": "five-class chest X-ray lesion ROI research classification",
        "input_scope": "pre-cropped ROI only; not a full image and not a detector",
        "predicted_class_id": predicted_class_id,
        "predicted_class_name_en": result["predicted_class_name"],
        "predicted_class_name_zh": CLASS_NAME_ZH[predicted_class_id],
        "softmax_confidence": float(result["confidence"]),
        "probabilities": probabilities,
        "ground_truth_class_id": ground_truth_id,
        "ground_truth_class_name_en": (
            result.get("ground_truth_class_name") if ground_truth_id is not None else None
        ),
        "ground_truth_class_name_zh": (
            CLASS_NAME_ZH[int(ground_truth_id)] if ground_truth_id is not None else None
        ),
        "is_correct": result.get("is_correct") if ground_truth_id is not None else None,
        "input_warnings": result.get("warnings", []),
        "model": {
            "architecture": result["architecture"],
            "initialization": result["initialization"],
            "model_sha256": result["model_sha256"],
        },
    }
    user_prompt = f"""請只根據下列 JSON 產生中文輔助說明，不可更改或重算任何數值：

```json
{json.dumps(structured, ensure_ascii=False, indent=2)}
```

請嚴格使用以下結構：

### 一、模型結果摘要
- 明確列出固定的預測類別（中英文）與 Softmax confidence。
- 若有 Ground Truth，可列出 Ground Truth 與 Correct/Incorrect；若無則寫「未提供 Ground Truth」。

### 二、類別一般說明
- 只提供預測類別的一般研究背景，最多 3 句。
- 不得描述這張 ROI 的影像內容、位置、嚴重程度或診斷。

### 三、機率與限制
- 解釋五類 Softmax 分數是相對分布，不是臨床機率。
- 可指出 class 2 胸膜增厚與 class 4 胸腔積液是本研究曾觀察到可能混淆的類別，但不得改判。

### 四、研究用途聲明
本結果僅供研究與系統展示，不是臨床診斷，不能取代合格醫療專業人員的判讀。
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, prompt_sha256(messages)
