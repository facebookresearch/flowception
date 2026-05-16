#!/usr/bin/env python3
"""Rule engine for Flowception installer metadata.

Loads JSON policy/matrix/monitoring files from this folder and evaluates
installer decisions for a given profile.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class EvaluationResult:
    triggered_rules: list[dict[str, Any]] = field(default_factory=list)
    blocked_rules: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    auto_fixes: dict[str, Any] = field(default_factory=dict)
    workflow_recommendations: list[dict[str, Any]] = field(default_factory=list)
    dependency_plan: dict[str, Any] = field(default_factory=dict)
    source_plan: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _intersects(lhs: list[str], rhs: list[str]) -> bool:
    return bool(set(lhs).intersection(rhs))


def _applies(applies_when: dict[str, Any], profile: dict[str, Any]) -> bool:
    if not applies_when:
        return True

    os_value = profile.get("os")
    if "os" in applies_when and os_value not in applies_when["os"]:
        return False

    arch_value = profile.get("arch")
    if "arch" in applies_when and arch_value not in applies_when["arch"]:
        return False

    selected_deps = profile.get("selectedDeps", [])
    if "selectedDepsAny" in applies_when and not _intersects(selected_deps, applies_when["selectedDepsAny"]):
        return False

    workflows = profile.get("workflow", [])
    if isinstance(workflows, str):
        workflows = [workflows]
    if "workflowAny" in applies_when and not _intersects(workflows, applies_when["workflowAny"]):
        return False

    model_variant = profile.get("modelVariant")
    if "modelVariantAny" in applies_when and model_variant not in applies_when["modelVariantAny"]:
        return False

    integration_intent = profile.get("integrationIntent")
    if "integrationIntent" in applies_when and integration_intent not in applies_when["integrationIntent"]:
        return False

    if "freeDiskTbLessThan" in applies_when:
        free_disk_tb = profile.get("freeDiskTb")
        if free_disk_tb is None or free_disk_tb >= applies_when["freeDiskTbLessThan"]:
            return False

    if "licenseAcceptanceRequired" in applies_when:
        required = applies_when["licenseAcceptanceRequired"]
        if bool(profile.get("licenseAcceptanceRequired", False)) != bool(required):
            return False

    if "licenseAccepted" in applies_when:
        if bool(profile.get("licenseAccepted", False)) != bool(applies_when["licenseAccepted"]):
            return False

    if "installStatus" in applies_when:
        status = profile.get("installStatus")
        if status not in applies_when["installStatus"]:
            return False

    for key, expected in applies_when.items():
        if key.startswith("userPreference."):
            pref_key = key.replace("userPreference.", "", 1)
            pref_val = _get_nested(profile, f"userPreference.{pref_key}")
            if pref_val != expected:
                return False

    return True


def _parse_set_action(action: str) -> tuple[str, Any] | None:
    if not action.startswith("set."):
        return None
    body = action.replace("set.", "", 1)
    if "=" not in body:
        return None
    key, value = body.split("=", 1)
    lowered = value.lower()
    if lowered in {"true", "false"}:
        parsed: Any = lowered == "true"
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
    return key, parsed


def evaluate_policy(policy: dict[str, Any], profile: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    auto_fixes: dict[str, Any] = {}

    for rule in policy.get("rules", []):
        if not _applies(rule.get("appliesWhen", {}), profile):
            continue

        findings.append(
            {
                "id": rule.get("id"),
                "severity": rule.get("severity"),
                "message": rule.get("message"),
                "autoRemediation": rule.get("autoRemediation", []),
                "confidence": rule.get("confidence", "medium"),
            }
        )

        for action in rule.get("autoRemediation", []):
            parsed = _parse_set_action(action)
            if parsed is not None:
                key, value = parsed
                auto_fixes[key] = value

    return findings, auto_fixes


def recommend_workflows(policy: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    hardware_profile = profile.get("hardwareProfile")

    for item in policy.get("workflows", {}).get("recommendedByHardware", []):
        target = item.get("hardwareProfile")
        if hardware_profile and target == hardware_profile:
            recommendations.append(item)

    return recommendations


def build_dependency_plan(matrix: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    os_value = profile.get("os")

    blocked: list[dict[str, Any]] = []
    for entry in matrix.get("decisionMatrix", {}).get("incompatible", []):
        blocked_when = entry.get("blockedWhen", {})
        blocked_oses = blocked_when.get("os", [])
        if blocked_oses and os_value in blocked_oses:
            blocked.append(entry)

    return {
        "essential": matrix.get("decisionMatrix", {}).get("essential", []),
        "optional": matrix.get("decisionMatrix", {}).get("optional", []),
        "experimental": matrix.get("decisionMatrix", {}).get("experimental", []),
        "blockedForProfile": blocked,
        "autoConfigProfiles": matrix.get("autoConfigProfiles", []),
    }


def build_source_plan(monitoring: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    sources = monitoring.get("topSources", [])
    source_feedback = profile.get("sourceFeedback", {})

    evaluated: list[dict[str, Any]] = []
    for src in sources:
        src_copy = dict(src)
        score = float(src_copy.get("score", 0.0))

        if source_feedback.get(src_copy.get("url")) == "not_helpful":
            score -= 0.25
            src_copy["confidence"] = "low"
            src_copy["label"] = monitoring.get("demotionAndLabeling", {}).get("lowConfidenceLabel", "Low Confidence")

        src_copy["effectiveScore"] = max(0.0, min(1.0, round(score, 3)))
        evaluated.append(src_copy)

    evaluated.sort(key=lambda x: x.get("effectiveScore", 0.0), reverse=True)
    target_count = monitoring.get("monitoringPolicy", {}).get("targetCount", 10)

    return {
        "topRanked": evaluated[:target_count],
        "allEvaluated": evaluated,
    }


def evaluate_profile(
    profile: dict[str, Any],
    policy_path: Path = BASE_DIR / "compatibility-policy.json",
    matrix_path: Path = BASE_DIR / "dependency-decision-matrix.json",
    monitoring_path: Path = BASE_DIR / "source-monitoring.json",
) -> EvaluationResult:
    policy = _load_json(policy_path)
    matrix = _load_json(matrix_path)
    monitoring = _load_json(monitoring_path)

    findings, auto_fixes = evaluate_policy(policy, profile)
    blocked = [f for f in findings if f.get("severity") == "hard_block"]
    warnings = [f for f in findings if f.get("severity") == "soft_warn"]

    return EvaluationResult(
        triggered_rules=findings,
        blocked_rules=blocked,
        warnings=warnings,
        auto_fixes=auto_fixes,
        workflow_recommendations=recommend_workflows(policy, profile),
        dependency_plan=build_dependency_plan(matrix, profile),
        source_plan=build_source_plan(monitoring, profile),
    )


def result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    return {
        "triggered_rules": result.triggered_rules,
        "blocked_rules": result.blocked_rules,
        "warnings": result.warnings,
        "auto_fixes": result.auto_fixes,
        "workflow_recommendations": result.workflow_recommendations,
        "dependency_plan": result.dependency_plan,
        "source_plan": result.source_plan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Flowception installer profile.")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Path to a JSON profile describing user/system choices.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON file path.",
    )
    args = parser.parse_args()

    profile_path = Path(args.profile)
    profile = _load_json(profile_path)

    result = evaluate_profile(profile)
    payload = result_to_dict(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    else:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
