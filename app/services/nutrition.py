"""
services/nutrition.py
Rule-based nutritional recommendation engine.
Maps predicted class index (0-3) → full guidance dict.
"""
from __future__ import annotations

from typing import Dict, Any


# ── Class index → severity label ──────────────────────────────────────────────
CLASS_NAMES: list[str] = ["Normal", "Mild", "Moderate", "Severe"]

# ── Short advice (used in compact responses) ──────────────────────────────────
nutrition_map: Dict[int, str] = {
    0: "Maintain a balanced diet with iron-rich foods.",
    1: "Increase iron and vitamin C intake.",
    2: "Iron, folate, and B12 required. Medical consultation recommended.",
    3: "Seek urgent medical care immediately.",
}

# ── Full guidance (used in detailed responses) ────────────────────────────────
_FULL_GUIDE: Dict[int, Dict[str, Any]] = {
    0: {
        "label": "Normal",
        "urgency": "Normal",
        "hb_range": "≥ 12 g/dL (F) | ≥ 13 g/dL (M)",
        "advice": "Maintain a balanced diet with iron-rich foods.",
        "foods": [
            "Lean red meat",
            "Poultry and fish",
            "Dark leafy greens (spinach, kale)",
            "Legumes (lentils, beans)",
            "Fortified cereals",
            "Citrus fruits (vitamin C boost)",
        ],
        "supplements": [],
        "referral": "No referral needed. Annual check-up recommended.",
        "followup_weeks": 52,
    },
    1: {
        "label": "Mild",
        "urgency": "Low",
        "hb_range": "10 – 11.9 g/dL",
        "advice": "Increase iron and vitamin C intake to enhance absorption.",
        "foods": [
            "Spinach (cooked)",
            "Lentils and dal",
            "Kidney and black beans",
            "Fortified breakfast cereals",
            "Dried apricots and raisins",
            "Bell peppers and citrus (vitamin C)",
            "Pumpkin seeds",
        ],
        "supplements": ["Low-dose iron supplement (consult physician)"],
        "referral": "Schedule physician visit within 4 weeks.",
        "followup_weeks": 8,
    },
    2: {
        "label": "Moderate",
        "urgency": "Medium",
        "hb_range": "7 – 9.9 g/dL",
        "advice": (
            "Structured dietary plan: iron, folate, and B12 required. "
            "Medical consultation strongly recommended."
        ),
        "foods": [
            "Liver (organ meat, high haem iron)",
            "Red meat 3–4 × per week",
            "Oysters and clams",
            "Dark leafy greens daily",
            "Fortified foods every morning",
            "Eggs",
            "Tofu (iron-fortified)",
            "Orange juice with every iron-rich meal",
        ],
        "supplements": [
            "Ferrous sulphate 200 mg (as prescribed)",
            "Vitamin C 500 mg with iron",
            "Folic acid 400 mcg daily",
        ],
        "referral": (
            "Medical consultation within 2 weeks. "
            "Request CBC, serum ferritin, serum iron, and TIBC."
        ),
        "followup_weeks": 4,
    },
    3: {
        "label": "Severe",
        "urgency": "Critical",
        "hb_range": "< 7 g/dL",
        "advice": (
            "URGENT: Immediate medical evaluation required. "
            "Diet alone is insufficient for severe anaemia."
        ),
        "foods": [
            "Iron-rich foods as tolerated under supervision",
            "Liver and organ meats",
            "Iron-fortified foods",
        ],
        "supplements": [
            "IV iron infusion (hospital-administered)",
            "Blood transfusion if clinically indicated",
        ],
        "referral": (
            "IMMEDIATE referral to emergency care or haematology. "
            "Risk of cardiac complications at HB < 7 g/dL."
        ),
        "followup_weeks": 1,
    },
}


# ── Binary (image-only) guidance — "Anemic" vs "Non-Anemic" ──────────────────
VISUAL_CLASS_NAMES: list[str] = ["Non-Anemic", "Anemic"]

_BINARY_GUIDE: dict[int, dict] = {
    0: _FULL_GUIDE[0],   # Non-Anemic — same as 4-class
    1: {
        "label": "Anemic",
        "urgency": "Moderate",
        "advice": (
            "Anemia detected from conjunctiva image. "
            "A full clinical assessment with hemoglobin measurement is recommended."
        ),
        "foods": [
            "Iron-rich foods: red meat, liver, spinach",
            "Vitamin C sources to improve absorption (citrus, bell peppers)",
            "Legumes: lentils, beans",
            "Fortified cereals and grains",
            "Eggs and dairy",
        ],
        "supplements": ["Consult a physician before starting iron supplements."],
        "referral": "Schedule a clinical evaluation with hemoglobin test to determine severity.",
        "followup_weeks": 4,
    },
}


def get_binary_guidance(class_idx: int) -> dict:
    """Return guidance for binary visual model (0=Non-Anemic, 1=Anemic)."""
    return dict(_BINARY_GUIDE.get(class_idx, _BINARY_GUIDE[1]))


# ── Public API ─────────────────────────────────────────────────────────────────
def get_short_advice(class_idx: int) -> str:
    """Return a single-line nutrition note for the predicted class."""
    return nutrition_map.get(class_idx, "Consult a healthcare professional.")


def get_full_guidance(
    class_idx: int,
    age_months: float | None = None,
    gender: int | None = None,
) -> Dict[str, Any]:
    """
    Return a full guidance dict for the predicted class.

    Optionally adjusts advice for infants (< 24 months) and females.
    """
    guide = dict(_FULL_GUIDE.get(class_idx, _FULL_GUIDE[0]))

    age_note = ""
    if age_months is not None:
        yrs = age_months / 12
        if yrs < 2:
            age_note = " (Infant: consult paediatrician before dietary changes.)"
        elif yrs < 12:
            age_note = " (Child: use age-appropriate iron sources.)"
        elif yrs > 60:
            age_note = " (Elderly: supplementation may improve absorption.)"

    gender_note = ""
    if gender == 1 and class_idx > 0:           # Female + anemic
        gender_note = " Females have elevated iron requirements."

    guide["advice"] = guide["advice"] + age_note + gender_note
    return guide
