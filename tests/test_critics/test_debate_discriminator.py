"""Unit tests for the 3-judge debate realism critic.

These tests do NOT call any LLM. They mock the judge factory to return
deterministic stub discriminators with pre-canned verdicts so we can
exercise: (a) the KS-statistic + unanimity computations, (b) the
majority-vote aggregation, (c) the KS-stopping early-halt path.
"""
from __future__ import annotations

import pytest

from attrforge.critics.debate_discriminator import (
    DebateConfig,
    DebateJudge,
    RealismDebate,
    ks_statistic_pairwise,
    majority_vote,
    unanimous_fraction,
)
from attrforge.critics.discriminator import (
    DiscriminationResult,
    DiscriminatorConfig,
    RealismDiscriminator,
)
from attrforge.schema import RealExample, RealismVerdict, SyntheticSample


def _real(t: str, idx: int) -> RealExample:
    return RealExample(text=t, label="general_question")


def _synth(t: str, idx: int) -> SyntheticSample:
    return SyntheticSample(
        sample_id=f"s{idx}",
        text=t,
        requested_attributes={"intent": "general_question"},
        generated_attributes={"intent": "general_question"},
        prompt_version=1,
        iteration=0,
    )


# ----- Pure-function tests ----------------------------------------------


def test_ks_statistic_zero_for_identical_judges():
    """When all judges produce the same verdict list, KS = 0."""
    preds = {
        "j1": ["real", "synthetic", "real", "synthetic"],
        "j2": ["real", "synthetic", "real", "synthetic"],
        "j3": ["real", "synthetic", "real", "synthetic"],
    }
    assert ks_statistic_pairwise(preds) == 0.0


def test_ks_statistic_positive_for_disagreeing_judges():
    """When judges disagree on the synthetic-fraction, KS > 0."""
    preds = {
        "j1": ["synthetic"] * 10,  # synth-fraction 1.0
        "j2": ["real"] * 10,  # synth-fraction 0.0
    }
    assert ks_statistic_pairwise(preds) == 1.0


def test_unanimous_fraction_all_agree():
    """When all judges vote the same per item, unanimity = 1.0."""
    preds = {
        "j1": ["real", "real", "synthetic"],
        "j2": ["real", "real", "synthetic"],
    }
    assert unanimous_fraction(preds) == 1.0


def test_unanimous_fraction_partial_disagreement():
    """One item disagrees -> unanimity = 2/3."""
    preds = {
        "j1": ["real", "real", "synthetic"],
        "j2": ["real", "synthetic", "synthetic"],
    }
    assert unanimous_fraction(preds) == 2 / 3


def test_majority_vote_breaks_ties_toward_real():
    """A 1-1 tie should resolve to 'real' (the conservative default)."""
    preds = {
        "j1": ["synthetic"],
        "j2": ["real"],
    }
    out = majority_vote(preds, ["s0"])
    assert out["s0"] == "real"


def test_majority_vote_majority_wins():
    """2 vs 1 -> majority class."""
    preds = {
        "j1": ["synthetic"],
        "j2": ["synthetic"],
        "j3": ["real"],
    }
    out = majority_vote(preds, ["s0"])
    assert out["s0"] == "synthetic"


# ----- Integration tests with mocked judges -----------------------------


def _stub_factory(canned_preds: list[list[str]]):
    """Build a judge factory that returns stub discriminators producing
    canned per-judge predictions in order."""
    call = {"i": 0}

    def factory(jcfg: DebateJudge) -> RealismDiscriminator:
        my_preds = canned_preds[call["i"]]
        call["i"] += 1

        class _StubDisc:
            def __init__(self):
                pass

            def judge(self, real, synthetic):
                # Build verdicts assuming the shuffle yields R000..Rk then synth.
                verdicts = []
                sids = [f"R{i:03d}" for i in range(len(real))] + [
                    s.sample_id for s in synthetic
                ]
                for sid, pred in zip(sids, my_preds):
                    verdicts.append(
                        RealismVerdict(
                            sample_id=sid,
                            prediction=pred,
                            confidence=0.5,
                            reason="stub",
                        )
                    )
                # Compute accuracy against ground truth (R000..Rk = real, rest synth).
                acc_correct = 0
                total = 0
                for sid, pred in zip(sids, my_preds):
                    truth = "real" if sid.startswith("R") else "synthetic"
                    if pred == truth:
                        acc_correct += 1
                    total += 1
                return DiscriminationResult(
                    verdicts=verdicts,
                    labels={
                        sid: ("real" if sid.startswith("R") else "synthetic")
                        for sid in sids
                    },
                    accuracy=acc_correct / total if total else 0.5,
                    synthetic_detection_rate=0.0,
                )

        return _StubDisc()

    return factory


def test_three_judge_debate_unanimous_short_circuits():
    """If all judges agree strongly early, KS-stopping halts before round 3."""
    # 4 real + 4 synthetic, all judges correctly identify everything.
    real = [_real(f"r{i}", i) for i in range(4)]
    synth = [_synth(f"s{i}", i) for i in range(4)]
    canned = [
        ["real"] * 4 + ["synthetic"] * 4,
        ["real"] * 4 + ["synthetic"] * 4,
        ["real"] * 4 + ["synthetic"] * 4,
    ]
    debate = RealismDebate(
        config=DebateConfig(
            judges=[
                DebateJudge(name="j1", model="x"),
                DebateJudge(name="j2", model="y"),
                DebateJudge(name="j3", model="z"),
            ],
            ks_threshold=0.10,
            unanimity_threshold=0.80,
            seed=17,
        ),
        judge_factory=_stub_factory(canned),
    )
    res = debate.judge(real, synth)
    # After 2 judges with full agreement, KS-stopping should fire.
    assert res.n_judges_called == 2
    assert res.stopped_early is True
    assert res.judge_agreement == 1.0
    assert res.ks_statistic == 0.0
    assert res.majority_accuracy == 1.0


def test_three_judge_debate_disagreement_runs_all_judges():
    """Heavy disagreement keeps the debate going through all three judges."""
    real = [_real(f"r{i}", i) for i in range(4)]
    synth = [_synth(f"s{i}", i) for i in range(4)]
    # Judges disagree wildly.
    canned = [
        ["real"] * 8,  # j1 calls everything real
        ["synthetic"] * 8,  # j2 calls everything synthetic
        ["real"] * 4 + ["synthetic"] * 4,  # j3 gets the right answer
    ]
    debate = RealismDebate(
        config=DebateConfig(
            judges=[
                DebateJudge(name="j1", model="x"),
                DebateJudge(name="j2", model="y"),
                DebateJudge(name="j3", model="z"),
            ],
            ks_threshold=0.10,
            unanimity_threshold=0.80,
            seed=17,
        ),
        judge_factory=_stub_factory(canned),
    )
    res = debate.judge(real, synth)
    assert res.n_judges_called == 3
    assert res.stopped_early is False
    # ks should be 1.0 between j1 and j2.
    assert res.ks_statistic >= 0.9
    # Per-judge accuracy: j1 = 0.5 (calls all real, half is right), j2 = 0.5, j3 = 1.0.
    assert res.per_judge_accuracy["j3"] == 1.0
