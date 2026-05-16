#!/usr/bin/env python3
"""Unit tests for the rule engine."""

from __future__ import annotations

import pytest

from rule_engine import (
    EvaluationResult,
    _applies,
    _intersects,
    _parse_set_action,
    evaluate_policy,
    evaluate_profile,
    result_to_dict,
)


class TestIntersects:

    def test_empty_lists(self) -> None:
        assert not _intersects([], [])

    def test_no_overlap(self) -> None:
        assert not _intersects(["a", "b"], ["c", "d"])

    def test_with_overlap(self) -> None:
        assert _intersects(["a", "b"], ["b", "c"])

    def test_single_element(self) -> None:
        assert _intersects(["x"], ["x"])


class TestParseSetAction:

    def test_valid_int(self) -> None:
        key, value = _parse_set_action("set.numInferenceSteps=40")
        assert key == "numInferenceSteps"
        assert value == 40
        assert isinstance(value, int)

    def test_valid_float(self) -> None:
        key, value = _parse_set_action("set.guidanceScale=1.5")
        assert key == "guidanceScale"
        assert abs(value - 1.5) < 0.01

    def test_valid_bool_true(self) -> None:
        key, value = _parse_set_action("set.enabled=true")
        assert key == "enabled"
        assert value is True

    def test_valid_bool_false(self) -> None:
        key, value = _parse_set_action("set.disabled=false")
        assert key == "disabled"
        assert value is False

    def test_string_value(self) -> None:
        key, value = _parse_set_action("set.model=distilled")
        assert key == "model"
        assert value == "distilled"

    def test_invalid_prefix(self) -> None:
        result = _parse_set_action("get.something=value")
        assert result is None

    def test_invalid_format(self) -> None:
        result = _parse_set_action("set.no_equals")
        assert result is None


class TestApplies:

    def test_empty_applies_when(self) -> None:
        assert _applies({}, {"os": "macos"})

    def test_os_match(self) -> None:
        profile = {"os": "macos"}
        assert _applies({"os": ["macos"]}, profile)
        assert not _applies({"os": ["linux"]}, profile)

    def test_arch_match(self) -> None:
        profile = {"arch": "arm64"}
        assert _applies({"arch": ["arm64"]}, profile)
        assert not _applies({"arch": ["x86_64"]}, profile)

    def test_selected_deps_any(self) -> None:
        profile = {"selectedDeps": ["pytorch", "flash-attn"]}
        assert _applies({"selectedDepsAny": ["flash-attn", "apex"]}, profile)
        assert not _applies({"selectedDepsAny": ["apex", "xformers"]}, profile)

    def test_workflow_any(self) -> None:
        profile = {"workflow": "openvid-i2v"}
        assert _applies({"workflowAny": ["openvid-i2v", "toy-i2v"]}, profile)
        assert not _applies({"workflowAny": ["other"]}, profile)

    def test_model_variant_any(self) -> None:
        profile = {"modelVariant": "distilled"}
        assert _applies({"modelVariantAny": ["distilled", "base"]}, profile)
        assert not _applies({"modelVariantAny": ["base"]}, profile)

    def test_free_disk_tb_less_than(self) -> None:
        profile = {"freeDiskTb": 1.5}
        assert _applies({"freeDiskTbLessThan": 5}, profile)
        assert not _applies({"freeDiskTbLessThan": 1.0}, profile)

    def test_license_accepted(self) -> None:
        profile = {"licenseAccepted": True}
        assert _applies({"licenseAccepted": True}, profile)
        assert not _applies({"licenseAccepted": False}, profile)

    def test_user_preference(self) -> None:
        profile = {"userPreference": {"autoRemoveFailedComponents": True}}
        applies_when = {"userPreference.autoRemoveFailedComponents": True}
        assert _applies(applies_when, profile)

        applies_when_false = {"userPreference.autoRemoveFailedComponents": False}
        assert not _applies(applies_when_false, profile)


class TestEvaluatePolicy:

    def test_no_matching_rules(self) -> None:
        policy = {
            "rules": [
                {
                    "id": "test-rule",
                    "severity": "hard_block",
                    "appliesWhen": {"os": ["linux"]},
                    "message": "This is Linux only",
                }
            ]
        }
        profile = {"os": "macos"}
        findings, auto_fixes = evaluate_policy(policy, profile)
        assert len(findings) == 0
        assert len(auto_fixes) == 0

    def test_matching_rule_no_remediation(self) -> None:
        policy = {
            "rules": [
                {
                    "id": "test-rule",
                    "severity": "soft_warn",
                    "appliesWhen": {"os": ["macos"]},
                    "message": "macOS warning",
                    "autoRemediation": [],
                }
            ]
        }
        profile = {"os": "macos"}
        findings, auto_fixes = evaluate_policy(policy, profile)
        assert len(findings) == 1
        assert findings[0]["id"] == "test-rule"
        assert len(auto_fixes) == 0

    def test_matching_rule_with_remediation(self) -> None:
        policy = {
            "rules": [
                {
                    "id": "test-rule",
                    "severity": "auto_fix",
                    "appliesWhen": {"modelVariant": "distilled"},
                    "message": "Setting distilled defaults",
                    "autoRemediation": ["set.guidanceScale=1.0", "set.numInferenceSteps=8"],
                }
            ]
        }
        profile = {"modelVariant": "distilled"}
        findings, auto_fixes = evaluate_policy(policy, profile)
        assert len(findings) == 1
        assert auto_fixes["guidanceScale"] == 1.0
        assert auto_fixes["numInferenceSteps"] == 8


class TestResultToDict:

    def test_conversion(self) -> None:
        result = EvaluationResult(
            triggered_rules=[{"id": "rule1", "severity": "warn"}],
            blocked_rules=[],
            warnings=[],
            auto_fixes={"key": "value"},
            workflow_recommendations=[],
            dependency_plan={"essential": []},
            source_plan={"topRanked": []},
        )
        payload = result_to_dict(result)
        assert "triggered_rules" in payload
        assert "auto_fixes" in payload
        assert payload["auto_fixes"]["key"] == "value"


class TestEvaluateProfile:

    def test_with_sample_profile(self) -> None:
        """Test evaluation with a minimal profile."""
        profile = {
            "os": "macos",
            "arch": "arm64",
            "hardwareProfile": "macbook-apple-silicon-laptop",
            "workflow": "toy-i2v",
            "selectedDeps": [],
            "modelVariant": "distilled",
        }

        result = evaluate_profile(profile)
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.triggered_rules, list)
        assert isinstance(result.blocked_rules, list)
        assert isinstance(result.auto_fixes, dict)
        assert isinstance(result.dependency_plan, dict)
        assert isinstance(result.source_plan, dict)

    def test_blocked_rule_detection(self) -> None:
        """Test that blockers are properly categorized."""
        profile = {
            "os": "macos",
            "arch": "arm64",
            "selectedDeps": ["flash-attn"],
            "licenseAcceptanceRequired": True,
            "licenseAccepted": False,
        }

        result = evaluate_profile(profile)
        assert len(result.blocked_rules) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
