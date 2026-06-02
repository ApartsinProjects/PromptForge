"""3-judge Debate Realism Critic with Kolmogorov-Smirnov adaptive stopping.

Implements scout D3.1 (arXiv:2510.12697): three LLM judges from different
model families produce independent realism verdicts on the shuffled mixed
batch, and a Kolmogorov-Smirnov adaptive stopping rule decides when to
stop sampling additional judge calls vs continue to a fourth round of
deliberation.

The three default judges (configurable):
    - openai/gpt-4o-mini       (OpenAI family)
    - anthropic/claude-3-haiku (Anthropic family)
    - google/gemini-flash-1.5  (Google family)

All three are served through OpenRouter (https://openrouter.ai/api/v1)
with a single OPENROUTER_API_KEY env var. The existing SynSmith LLMClient
already supports a custom base_url, so this critic just wires three of
them at different model strings.

Outputs (DebateResult):
    - per_judge_accuracy: dict[judge_name, accuracy]
    - majority_accuracy:  per-sample majority verdict accuracy
    - judge_agreement:    fraction of samples where >=2 judges agree
    - ks_statistic:       max difference between any two judges'
                          synthetic-vs-real verdict distributions
    - stopped_early:      True if KS-stopping halted early (all judges
                          agree closely enough that more deliberation is
                          unlikely to change the verdict)

Replaces the standard RealismDiscriminator in the `full_attrforge_3judge`
baseline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from synsmith.critics.discriminator import (
    DiscriminationResult,
    DiscriminatorConfig,
    RealismDiscriminator,
)
from synsmith.llm import LLMClient, LLMConfig
from synsmith.schema import RealExample, RealismVerdict, SyntheticSample


@dataclass
class DebateJudge:
    name: str
    model: str
    backend: str = "openai"  # OpenRouter is OpenAI-compatible
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 600


@dataclass
class DebateConfig:
    judges: list[DebateJudge] = field(
        default_factory=lambda: [
            DebateJudge(
                name="gpt-4o-mini",
                model="openai/gpt-4o-mini",
            ),
            DebateJudge(
                name="claude-3-haiku",
                model="anthropic/claude-3-haiku",
            ),
            DebateJudge(
                name="gemini-flash-1.5",
                model="google/gemini-flash-1.5",
            ),
        ]
    )
    # KS-stopping: when the maximum per-sample KS distance between any two
    # judges' verdict distributions falls below this threshold AND at least
    # this fraction of samples have unanimous verdicts, stop early instead of
    # spending budget on additional rounds.
    ks_threshold: float = 0.10
    unanimity_threshold: float = 0.80
    max_samples: int = 24
    seed: int | None = None


@dataclass
class DebateResult:
    """Aggregated verdict + per-judge breakdown + KS-stopping diagnostics."""

    verdicts: list[RealismVerdict]
    per_judge_accuracy: dict[str, float]
    majority_accuracy: float
    judge_agreement: float
    ks_statistic: float
    stopped_early: bool
    n_judges_called: int


def ks_statistic_pairwise(
    judge_predictions: dict[str, list[str]],
) -> float:
    """Max difference between any two judges' synthetic-fraction.

    The synthetic-fraction is the proportion of samples a judge labeled
    "synthetic". KS-style: the larger the spread across judges, the more
    deliberation is needed; the smaller the spread, the safer to stop.
    """
    if not judge_predictions:
        return 0.0
    fractions = {}
    for name, preds in judge_predictions.items():
        if not preds:
            continue
        fractions[name] = sum(1 for p in preds if p == "synthetic") / len(preds)
    if len(fractions) < 2:
        return 0.0
    vals = list(fractions.values())
    return float(max(vals) - min(vals))


def unanimous_fraction(
    judge_predictions: dict[str, list[str]],
) -> float:
    """Fraction of samples where all judges agree on the label."""
    if not judge_predictions:
        return 0.0
    n_samples_list = [len(v) for v in judge_predictions.values() if v]
    if not n_samples_list:
        return 0.0
    n_samples = min(n_samples_list)
    if n_samples == 0:
        return 0.0
    n_unanimous = 0
    for i in range(n_samples):
        verdicts = {preds[i] for preds in judge_predictions.values() if i < len(preds)}
        if len(verdicts) == 1:
            n_unanimous += 1
    return n_unanimous / n_samples


def majority_vote(
    judge_predictions: dict[str, list[str]],
    sample_ids: Sequence[str],
) -> dict[str, str]:
    """Majority-vote across judges, per sample id. Ties broken toward 'real'."""
    out: dict[str, str] = {}
    if not judge_predictions or not sample_ids:
        return out
    for i, sid in enumerate(sample_ids):
        votes: dict[str, int] = {}
        for preds in judge_predictions.values():
            if i < len(preds):
                votes[preds[i]] = votes.get(preds[i], 0) + 1
        if not votes:
            continue
        top = max(votes.values())
        winners = [k for k, v in votes.items() if v == top]
        if "real" in winners:
            out[sid] = "real"
        else:
            out[sid] = winners[0]
    return out


class RealismDebate:
    """3-judge debate-style realism critic.

    Each judge is an independent RealismDiscriminator wrapping a different
    LLM. Per round, every judge returns its own verdict on the same shuffled
    batch; the KS-stopping rule decides whether to halt after the first round
    (most cases) or call additional judges (borderline cases).
    """

    def __init__(
        self,
        config: DebateConfig | None = None,
        judge_factory=None,
    ) -> None:
        self.config = config or DebateConfig()
        self._rng = random.Random(self.config.seed)
        # Allow dependency injection of a judge factory for tests.
        if judge_factory is None:
            self._judge_factory = self._default_judge_factory
        else:
            self._judge_factory = judge_factory

    def _default_judge_factory(self, judge: DebateJudge) -> RealismDiscriminator:
        """Construct a RealismDiscriminator wired to one judge LLM."""
        llm_cfg = LLMConfig(
            backend=judge.backend,
            model=judge.model,
            api_key_env=judge.api_key_env,
            base_url=judge.base_url,
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
        client = LLMClient(llm_cfg)
        return RealismDiscriminator(
            client=client,
            config=DiscriminatorConfig(
                max_samples=self.config.max_samples,
                temperature=judge.temperature,
                seed=self.config.seed,
            ),
        )

    def judge(
        self,
        real: list[RealExample],
        synthetic: list[SyntheticSample],
    ) -> DebateResult:
        """Run all judges on the same shuffled batch; KS-stop if they agree."""
        # Pre-stage: get a single shuffled batch + ground-truth labels so
        # every judge votes on the same samples in the same order.
        labels: dict[str, str] = {}
        mixed: list[tuple[str, str]] = []
        for i, ex in enumerate(real):
            sid = f"R{i:03d}"
            labels[sid] = "real"
            mixed.append((sid, ex.text))
        for s in synthetic:
            labels[s.sample_id] = "synthetic"
            mixed.append((s.sample_id, s.text))
        self._rng.shuffle(mixed)
        mixed = mixed[: self.config.max_samples]
        sample_ids = [sid for sid, _ in mixed]
        # Build per-judge in stub form sharing the same shuffled batch.
        # We bypass each judge's internal shuffle by feeding pre-shuffled
        # data directly. To keep this minimally invasive, we drive each
        # judge's full judge() method on the same `real`+`synthetic` lists
        # and set its rng seed so its shuffle reproduces.
        judge_preds: dict[str, list[str]] = {}
        per_judge_acc: dict[str, float] = {}
        n_called = 0
        for jcfg in self.config.judges:
            disc = self._judge_factory(jcfg)
            n_called += 1
            res = disc.judge(real, synthetic)
            # Sort verdicts by sample_id so cross-judge alignment is stable.
            preds_by_sid = {v.sample_id: v.prediction for v in res.verdicts}
            ordered_preds = [
                preds_by_sid.get(sid, "real") for sid in sample_ids
            ]
            judge_preds[jcfg.name] = ordered_preds
            per_judge_acc[jcfg.name] = res.accuracy
            # KS-stopping check after each judge call (from the second onward).
            if n_called >= 2:
                ks = ks_statistic_pairwise(judge_preds)
                ufrac = unanimous_fraction(judge_preds)
                if (
                    ks <= self.config.ks_threshold
                    and ufrac >= self.config.unanimity_threshold
                    and n_called < len(self.config.judges)
                ):
                    # Strong agreement; skip remaining judges this round.
                    break

        majority = majority_vote(judge_preds, sample_ids)
        verdicts = [
            RealismVerdict(
                sample_id=sid,
                prediction=majority.get(sid, "real"),
                confidence=0.5,
                reason="3-judge debate (majority vote)",
            )
            for sid in sample_ids
        ]
        if labels and majority:
            correct = sum(
                1 for sid, pred in majority.items() if labels.get(sid) == pred
            )
            majority_acc = correct / len(majority)
        else:
            majority_acc = 0.5

        ks = ks_statistic_pairwise(judge_preds)
        ufrac = unanimous_fraction(judge_preds)
        return DebateResult(
            verdicts=verdicts,
            per_judge_accuracy=per_judge_acc,
            majority_accuracy=majority_acc,
            judge_agreement=ufrac,
            ks_statistic=ks,
            stopped_early=(n_called < len(self.config.judges)),
            n_judges_called=n_called,
        )
