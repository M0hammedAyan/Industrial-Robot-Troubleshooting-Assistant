"""Tests for grounding / hallucination controls."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.grounding import (
    evidence_supports_question,
    extractive_error_answer,
    should_refuse_generation,
)


def test_refuse_robot_xyz():
    chunks = [
        {
            "text": "Maximum recommended tool mass for IR-7000-14 including gripper: 14 kg.",
            "source": "robot_user_manual.pdf",
            "page": 4,
        }
    ]
    ok, reason = evidence_supports_question(
        "What is the maximum payload of Robot XYZ?", chunks
    )
    assert not ok
    assert reason == "unsupported_entity"
    refuse, reason = should_refuse_generation(
        "What is the maximum payload of Robot XYZ?", chunks, 0.43, 0.35
    )
    assert refuse
    assert reason == "unsupported_entity"


def test_extractive_error_code():
    chunks = [
        {
            "text": (
                "E-123 Analog input out of range\n"
                "Meaning: Configured analog input channel exceeded high/low limits.\n"
                "Possible causes: Sensor failure; wiring open/short."
            ),
            "source": "troubleshooting_manual.pdf",
            "page": 1,
        }
    ]
    answer = extractive_error_answer("What does error E-123 mean?", chunks)
    assert answer is not None
    assert "Analog input out of range" in answer
    assert "Invalid operating mode" not in answer
