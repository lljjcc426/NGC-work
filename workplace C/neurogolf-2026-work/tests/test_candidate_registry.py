from __future__ import annotations

from candidate_registry import risk_delta, strict_record_errors


def _runtime_safe_record() -> dict:
    return {
        "task": "task001",
        "status": "runtime_safe",
        "official_examples_checked": 266,
        "official_examples_passed": 266,
        "fuzz_trials": 2000,
        "generator_trials": 0,
        "runtime_compatible": True,
        "inherited_parent_risk": False,
        "introduced_runtime_risks": [],
        "introduced_negative_padding": [],
        "parent_cost": 100,
        "cost": 90,
    }


def test_runtime_safe_requires_full_evidence() -> None:
    record = _runtime_safe_record()
    assert strict_record_errors(record) == []

    record["official_examples_passed"] = 265
    record["fuzz_trials"] = 10
    assert strict_record_errors(record) == [
        "official_examples_incomplete",
        "insufficient_exact_fuzz",
    ]


def test_runtime_safe_requires_real_cost_gain() -> None:
    record = _runtime_safe_record()
    record["cost"] = 100
    assert strict_record_errors(record) == ["cost_not_lower_than_parent"]


def test_online_verified_requires_complete_reference() -> None:
    record = _runtime_safe_record()
    record.update({
        "status": "online_verified",
        "online_status": "COMPLETE",
        "online_kaggle_ref": 12345,
        "online_public_score": 7381.68,
        "online_package_delta": 0.01,
    })
    assert strict_record_errors(record) == []
    record["online_status"] = "ERROR"
    assert strict_record_errors(record) == ["online_status_not_complete"]


def test_risk_delta_allows_inherited_parent_risk() -> None:
    parent = {
        "risk_fingerprint": {
            "runtime": ["TopK:int8"],
            "negative_padding": ["[-1, 0, -1, 0]"],
        }
    }
    candidate = {
        "risk_fingerprint": {
            "runtime": ["TopK:int8"],
            "negative_padding": ["[-1, 0, -1, 0]"],
        }
    }
    assert risk_delta(parent, candidate) == {"runtime": [], "negative_padding": []}


def test_risk_delta_blocks_new_dtype_or_padding_risk() -> None:
    parent = {"risk_fingerprint": {"runtime": [], "negative_padding": []}}
    candidate = {
        "risk_fingerprint": {
            "runtime": ["ScatterND:int32"],
            "negative_padding": ["[-1, 0, -1, 0]"],
        }
    }
    assert risk_delta(parent, candidate) == {
        "runtime": ["ScatterND:int32"],
        "negative_padding": ["[-1, 0, -1, 0]"],
    }
