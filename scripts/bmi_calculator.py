"""
BMI Calculator Module
Provides BMI calculation, categorization, and health recommendations.
"""


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """
    Calculate Body Mass Index (BMI).

    Args:
        weight_kg: Weight in kilograms.
        height_cm: Height in centimeters.

    Returns:
        BMI value rounded to 1 decimal place.

    Raises:
        ValueError: If inputs are invalid (non-positive).
    """
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("Weight and height must be positive numbers.")

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)


def get_bmi_category(bmi: float) -> str:
    """
    Classify BMI into WHO categories.

    Categories:
        - Underweight:  BMI < 18.5
        - Normal:       18.5 <= BMI < 25
        - Overweight:   25 <= BMI < 30
        - Obese:        BMI >= 30
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def get_bmi_color(bmi: float) -> str:
    """Return a hex color for BMI category (for UI display)."""
    if bmi < 18.5:
        return "#5dade2"  # Steel blue
    elif bmi < 25:
        return "#48c9b0"  # Teal green
    elif bmi < 30:
        return "#f0b27a"  # Muted amber
    else:
        return "#cd6155"  # Dusty red


def get_health_recommendation(bmi: float) -> str:
    """Return a health recommendation based on BMI category."""
    category = get_bmi_category(bmi)
    recommendations = {
        "Underweight": (
            "Consider increasing calorie intake with nutrient-dense foods. "
            "Consult a healthcare provider to rule out underlying conditions."
        ),
        "Normal": (
            "Maintain your current healthy weight through balanced nutrition "
            "and regular physical activity."
        ),
        "Overweight": (
            "Consider adopting a healthier diet and increasing physical activity. "
            "Aim for 150 minutes of moderate exercise per week."
        ),
        "Obese": (
            "Consult a healthcare provider for a personalized weight management plan. "
            "Focus on sustainable lifestyle changes."
        ),
    }
    return recommendations[category]


def bmi_full_report(weight_kg: float, height_cm: float) -> dict:
    """
    Generate a complete BMI report.

    Returns:
        Dictionary with bmi, category, color, and recommendation.
    """
    bmi = calculate_bmi(weight_kg, height_cm)
    return {
        "bmi": bmi,
        "category": get_bmi_category(bmi),
        "color": get_bmi_color(bmi),
        "recommendation": get_health_recommendation(bmi),
    }


if __name__ == "__main__":
    # Demo
    print("=== BMI Calculator Demo ===\n")
    test_cases = [
        (45, 170, "Underweight"),  # BMI ~15.6
        (70, 170, "Normal"),       # BMI ~24.2
        (85, 170, "Overweight"),   # BMI ~29.4
        (110, 170, "Obese"),       # BMI ~38.1
    ]
    for weight, height, expected in test_cases:
        report = bmi_full_report(weight, height)
        status = "OK" if report["category"] == expected else "FAIL"
        print(f"[{status}] {weight}kg / {height}cm -> "
              f"BMI {report['bmi']} ({report['category']})")
