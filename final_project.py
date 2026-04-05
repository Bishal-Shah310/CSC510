import math
import json
from typing import List, Dict, Tuple


# -----------------------------
# Sample historical case data
# -----------------------------
# This acts like a tiny knowledge base for "searching" similar cases.
PAST_CASES = [
    {
        "case_id": 1,
        "age": 22,
        "claims": 4,
        "policy_type": "auto",
        "location_risk": "high",
        "risk": "High",
        "decision": "Manual Review"
    },
    {
        "case_id": 2,
        "age": 45,
        "claims": 0,
        "policy_type": "home",
        "location_risk": "low",
        "risk": "Low",
        "decision": "Approve"
    },
    {
        "case_id": 3,
        "age": 31,
        "claims": 2,
        "policy_type": "business",
        "location_risk": "medium",
        "risk": "Medium",
        "decision": "Request More Information"
    },
    {
        "case_id": 4,
        "age": 27,
        "claims": 1,
        "policy_type": "auto",
        "location_risk": "medium",
        "risk": "Medium",
        "decision": "Request More Information"
    },
    {
        "case_id": 5,
        "age": 54,
        "claims": 5,
        "policy_type": "business",
        "location_risk": "high",
        "risk": "High",
        "decision": "Manual Review"
    },
    {
        "case_id": 6,
        "age": 38,
        "claims": 0,
        "policy_type": "auto",
        "location_risk": "low",
        "risk": "Low",
        "decision": "Approve"
    }
]


VALID_POLICY_TYPES = {"auto", "home", "business"}
VALID_LOCATION_RISKS = {"low", "medium", "high"}


# -----------------------------
# Input helpers
# -----------------------------
def get_int_input(prompt: str, min_value: int = None, max_value: int = None) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
            if min_value is not None and value < min_value:
                print(f"Please enter a value greater than or equal to {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Please enter a value less than or equal to {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a whole number.")


def get_choice_input(prompt: str, valid_choices: set) -> str:
    while True:
        value = input(prompt).strip().lower()
        if value in valid_choices:
            return value
        print(f"Invalid input. Choose one of: {', '.join(sorted(valid_choices))}")


# -----------------------------
# Classification
# -----------------------------
# This is a simple logistic-style risk scoring function.
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def calculate_risk_probability(age: int, claims: int, policy_type: str, location_risk: str) -> float:
    score = -2.0

    # Age factor
    if age < 25:
        score += 1.4
    elif age < 35:
        score += 0.6
    elif age > 60:
        score += 0.5

    # Claims factor
    score += claims * 0.7

    # Policy type factor
    if policy_type == "business":
        score += 0.8
    elif policy_type == "auto":
        score += 0.4
    elif policy_type == "home":
        score += 0.2

    # Location risk factor
    if location_risk == "high":
        score += 1.0
    elif location_risk == "medium":
        score += 0.5

    return sigmoid(score)


def classify_risk(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    elif probability >= 0.45:
        return "Medium"
    return "Low"


# -----------------------------
# Expert system rules
# -----------------------------
# These are deterministic rules that explain the decision.
def apply_expert_rules(age: int, claims: int, policy_type: str, location_risk: str, base_risk: str) -> Tuple[str, List[str]]:
    reasons = []

    if age < 25:
        reasons.append("Applicant is under 25, which increases risk.")
    if claims >= 3:
        reasons.append("Claim history is high.")
    elif claims >= 1:
        reasons.append("Applicant has prior claim activity.")

    if location_risk == "high":
        reasons.append("Location is considered high risk.")
    elif location_risk == "medium":
        reasons.append("Location is considered medium risk.")

    if policy_type == "business":
        reasons.append("Business policies usually require additional review.")

    # Rule-based decision
    if base_risk == "High" and location_risk == "high":
        decision = "Manual Review"
        reasons.append("High-risk classification combined with high-risk location triggered manual review.")
    elif claims >= 4:
        decision = "Manual Review"
        reasons.append("Very high number of claims triggered manual review.")
    elif base_risk == "Medium":
        decision = "Request More Information"
        reasons.append("Medium-risk classification triggered additional validation.")
    else:
        decision = "Approve"
        reasons.append("No critical rules were triggered, so the case can be approved.")

    return decision, reasons


# -----------------------------
# Search method
# -----------------------------
# This searches the case base for the most similar historical example.
def encode_location_risk(location_risk: str) -> int:
    mapping = {"low": 1, "medium": 2, "high": 3}
    return mapping[location_risk]


def encode_policy_type(policy_type: str) -> int:
    mapping = {"home": 1, "auto": 2, "business": 3}
    return mapping[policy_type]


def calculate_similarity_score(user_case: Dict, past_case: Dict) -> float:
    age_diff = abs(user_case["age"] - past_case["age"])
    claims_diff = abs(user_case["claims"] - past_case["claims"])
    policy_diff = abs(encode_policy_type(user_case["policy_type"]) - encode_policy_type(past_case["policy_type"]))
    location_diff = abs(encode_location_risk(user_case["location_risk"]) - encode_location_risk(past_case["location_risk"]))

    # Lower score means more similar
    return (age_diff * 0.2) + (claims_diff * 1.5) + (policy_diff * 1.0) + (location_diff * 1.2)


def find_most_similar_case(user_case: Dict, past_cases: List[Dict]) -> Dict:
    best_case = None
    best_score = float("inf")

    for case in past_cases:
        score = calculate_similarity_score(user_case, case)
        if score < best_score:
            best_score = score
            best_case = case

    return best_case


# -----------------------------
# Symbolic planning
# -----------------------------
# This returns recommended next actions based on the outcome.
def build_action_plan(decision: str, risk: str) -> List[str]:
    if decision == "Approve":
        return [
            "Validate final applicant details",
            "Generate approval notice",
            "Move application to approved status"
        ]
    elif decision == "Request More Information":
        return [
            "Request supporting documents from applicant",
            "Re-evaluate claim history and policy details",
            "Submit updated application for underwriting review"
        ]
    elif decision == "Manual Review":
        return [
            "Escalate case to underwriting specialist",
            "Review claim history, policy type, and location factors",
            "Make final human decision and document rationale"
        ]
    else:
        return [
            "Hold application",
            "Review case manually",
            "Record final outcome"
        ]


# -----------------------------
# Reporting
# -----------------------------
def print_report(report: Dict) -> None:
    print("\n" + "=" * 60)
    print("AI INSURANCE RISK DECISION TOOL - FINAL REPORT")
    print("=" * 60)

    print("\nApplicant Input")
    print(f"Age: {report['input']['age']}")
    print(f"Number of Claims: {report['input']['claims']}")
    print(f"Policy Type: {report['input']['policy_type']}")
    print(f"Location Risk: {report['input']['location_risk']}")

    print("\nClassification Result")
    print(f"Risk Probability: {report['classification']['probability']:.2f}")
    print(f"Risk Level: {report['classification']['risk_level']}")

    print("\nExpert System Decision")
    print(f"Decision: {report['decision']}")
    print("Reasons:")
    for reason in report["reasons"]:
        print(f"- {reason}")

    print("\nMost Similar Historical Case")
    similar = report["similar_case"]
    print(f"Case ID: {similar['case_id']}")
    print(f"Age: {similar['age']}")
    print(f"Claims: {similar['claims']}")
    print(f"Policy Type: {similar['policy_type']}")
    print(f"Location Risk: {similar['location_risk']}")
    print(f"Historical Risk: {similar['risk']}")
    print(f"Historical Decision: {similar['decision']}")

    print("\nRecommended Action Plan")
    for step_number, step in enumerate(report["action_plan"], start=1):
        print(f"{step_number}. {step}")

    print("=" * 60)


def save_report_to_json(report: Dict, filename: str = "insurance_risk_report.json") -> None:
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)
        print(f"\nReport saved successfully to {filename}")
    except OSError as exc:
        print(f"\nCould not save report: {exc}")


# -----------------------------
# Main program
# -----------------------------
def main() -> None:
    print("=" * 60)
    print("WELCOME TO THE AI INSURANCE RISK DECISION TOOL")
    print("=" * 60)
    print("This program uses:")
    print("- Classification")
    print("- Rule-based expert system logic")
    print("- Search for similar historical cases")
    print("- Symbolic planning for next actions")
    print("=" * 60)

    age = get_int_input("Enter applicant age: ", min_value=18, max_value=100)
    claims = get_int_input("Enter number of prior claims: ", min_value=0, max_value=50)
    policy_type = get_choice_input("Enter policy type (auto/home/business): ", VALID_POLICY_TYPES)
    location_risk = get_choice_input("Enter location risk (low/medium/high): ", VALID_LOCATION_RISKS)

    probability = calculate_risk_probability(age, claims, policy_type, location_risk)
    risk_level = classify_risk(probability)
    decision, reasons = apply_expert_rules(age, claims, policy_type, location_risk, risk_level)

    user_case = {
        "age": age,
        "claims": claims,
        "policy_type": policy_type,
        "location_risk": location_risk
    }

    similar_case = find_most_similar_case(user_case, PAST_CASES)
    action_plan = build_action_plan(decision, risk_level)

    report = {
        "input": user_case,
        "classification": {
            "probability": probability,
            "risk_level": risk_level
        },
        "decision": decision,
        "reasons": reasons,
        "similar_case": similar_case,
        "action_plan": action_plan
    }

    print_report(report)

    save_choice = input("\nWould you like to save this report as JSON? (yes/no): ").strip().lower()
    if save_choice in {"yes", "y"}:
        save_report_to_json(report)

    print("\nProgram finished successfully.")


if __name__ == "__main__":
    main()