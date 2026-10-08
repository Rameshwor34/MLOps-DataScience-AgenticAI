import pytest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_golden_dataset_loads():
    dataset_path = ROOT / "eval" / "golden_dataset.json"
    assert dataset_path.exists()

def test_golden_dataset_has_required_fields():
    dataset_path = ROOT / "eval" / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    for item in data:
        assert "id" in item
        assert "query" in item
        assert "reference_response" in item
        assert "category" in item

def test_regression_threshold_logic():
    passed = 4
    total = 5
    pct_passed = passed / total
    assert pct_passed == 0.8
    # threshold is 0.80
    assert pct_passed >= 0.80
