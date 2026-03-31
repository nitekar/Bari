from __future__ import annotations

from typing import Literal

RiskLevel = Literal["low", "moderate", "high"]

BASE_FOODS_INCLUDE = [
    "Liver",
    "Red meat",
    "Beans and lentils",
    "Spinach",
    "Citrus fruits (oranges, lemons)",
    "Tomatoes",
]

FOODS_TO_AVOID = [
    "Tea during or 1 hour around iron-rich meals",
    "Coffee during or 1 hour around iron-rich meals",
    "Calcium-heavy foods (milk, yoghurt, cheese) at the same time as iron supplements",
]


def severity_to_risk(prediction: str) -> RiskLevel:
    if prediction in ("Severe", "Anemic"):
        return "high"
    if prediction == "Moderate":
        return "moderate"
    return "low"


def build_structured_recommendations(
    prediction: str,
    confidence: float,
    age_months: float | None = None,
) -> dict:
    risk: RiskLevel = severity_to_risk(prediction)

    if risk == "low":
        diet_plan = (
            "Maintain a balanced diet with regular iron-rich foods. "
            "Include haem iron (meat) 2–3 times per week and plant iron every day."
        )
        urgency = "routine"
    elif risk == "moderate":
        diet_plan = (
            "Increase iron intake in every main meal and add vitamin C sources to boost absorption. "
            "Monitor the child and schedule a medical check within 4 weeks."
        )
        urgency = "elevated"
    else:
        diet_plan = (
            "URGENT: strong dietary intervention plus immediate referral. "
            "Diet alone is not sufficient — clinical assessment is required."
        )
        urgency = "urgent"

    if age_months is not None:
        if 6 <= age_months < 24:
            diet_plan += " For infants 6–24 months, prioritise breast milk or iron-fortified formula and consult a paediatrician."
        elif 24 <= age_months <= 60:
            diet_plan += " For children 2–5 years, use age-appropriate iron-rich solid foods and avoid choking hazards."

    foods_to_include = list(BASE_FOODS_INCLUDE)
    if risk != "low":
        foods_to_include += ["Fortified cereals", "Eggs"]

    return {
        "risk_level": risk,
        "confidence_score": float(confidence),
        "recommendations": {
            "diet_plan": diet_plan,
            "foods_to_include": foods_to_include,
            "foods_to_avoid": FOODS_TO_AVOID,
            "urgency_level": urgency,
        },
    }

