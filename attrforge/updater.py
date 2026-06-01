"""Prompt Updater.

Reads aggregated critic feedback and rewrites the generator prompt. Keeps
a versioned history of every prompt the loop produced, with the feedback
that motivated each update, so a researcher can later attribute behavioral
changes back to specific critic signals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from attrforge.llm import LLMClient
from attrforge.prompts import UPDATER_SYSTEM, UPDATER_USER_TEMPLATE
from attrforge.schema import (
    AttributeVerdict,
    DiversityReport,
    IterationFeedback,
    RealismVerdict,
)


@dataclass
class PromptVersion:
    """One entry in the prompt history."""

    version: int
    iteration: int
    prompt: str
    motivation: str
    feedback_summary: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PromptHistory:
    """Append-only log of prompt versions and the feedback that produced them."""

    def __init__(self, initial_prompt: str) -> None:
        self._versions: list[PromptVersion] = [
            PromptVersion(
                version=0,
                iteration=0,
                prompt=initial_prompt,
                motivation="initial prompt",
                feedback_summary={},
            )
        ]

    @property
    def current(self) -> PromptVersion:
        return self._versions[-1]

    @property
    def current_prompt(self) -> str:
        return self.current.prompt

    @property
    def current_version(self) -> int:
        return self.current.version

    def append(
        self,
        prompt: str,
        iteration: int,
        motivation: str,
        feedback_summary: dict,
    ) -> PromptVersion:
        pv = PromptVersion(
            version=self.current.version + 1,
            iteration=iteration,
            prompt=prompt,
            motivation=motivation,
            feedback_summary=feedback_summary,
        )
        self._versions.append(pv)
        return pv

    def to_list(self) -> list[dict]:
        return [
            {
                "version": v.version,
                "iteration": v.iteration,
                "prompt": v.prompt,
                "motivation": v.motivation,
                "feedback_summary": v.feedback_summary,
                "timestamp": v.timestamp,
            }
            for v in self._versions
        ]


class PromptUpdater:
    """Rewrite the generator prompt from a bundle of critic feedback."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_failures_in_prompt: int = 8,
        max_artifacts_in_prompt: int = 8,
    ) -> None:
        self.client = client
        self.max_failures = max_failures_in_prompt
        self.max_artifacts = max_artifacts_in_prompt

    def update(
        self,
        current_prompt: str,
        feedback: IterationFeedback,
    ) -> tuple[str, dict]:
        """Return ``(new_prompt, summary_for_history)``.

        ``summary_for_history`` is a compact dict that's safe to persist in
        the prompt history without bloating it with every raw verdict.
        """
        attribute_block = self._format_attribute_failures(feedback.attribute_failures)
        realism_block = self._format_realism_artifacts(feedback.realism_artifacts)
        diversity_block = self._format_diversity(feedback.diversity)
        pack_block = self._format_pack(feedback)
        mode_seeking_block = self._format_mode_seeking(feedback)
        banned_block = self._format_banned(feedback)
        coverage_hole_block = self._format_coverage_holes(feedback)

        user_msg = UPDATER_USER_TEMPLATE.format(
            current_prompt=current_prompt,
            attribute_block=attribute_block,
            realism_block=realism_block,
            diversity_block=diversity_block,
            pack_block=pack_block,
            mode_seeking_block=mode_seeking_block,
            banned_block=banned_block,
            coverage_hole_block=coverage_hole_block,
        )
        new_prompt = self.client.chat(
            UPDATER_SYSTEM,
            [{"role": "user", "content": user_msg}],
            temperature=0.3,
            max_tokens=700,
        ).strip()

        if new_prompt.startswith("```"):
            new_prompt = new_prompt.strip("`").lstrip("text\n").lstrip("\n").rstrip("`")

        summary = {
            "iteration": feedback.iteration,
            "n_attribute_failures": len(feedback.attribute_failures),
            "n_realism_artifacts": len(feedback.realism_artifacts),
            "near_duplicate_rate": feedback.diversity.near_duplicate_rate,
            "pack_accuracy": feedback.pack_accuracy,
            "mode_seeking_ratio": feedback.mode_seeking_ratio,
            "n_banned_phrasings": len(feedback.banned_phrasings),
            "missing_modes": feedback.diversity.missing_modes,
            "overrepresented_modes": feedback.diversity.overrepresented_modes,
            "metrics": feedback.metrics,
        }
        return new_prompt, summary

    def _format_pack(self, feedback: IterationFeedback) -> str:
        if feedback.pack_accuracy is None:
            return "(pack discriminator disabled)"
        lines = [f"pack_accuracy: {feedback.pack_accuracy:.2f} (chance = 0.50)"]
        if feedback.pack_artifacts:
            lines.append("repeated patterns across packs:")
            for p in feedback.pack_artifacts[:8]:
                lines.append(f"  - {p}")
        return "\n".join(lines)

    def _format_mode_seeking(self, feedback: IterationFeedback) -> str:
        if feedback.mode_seeking_ratio is None:
            return "(mode-seeking disabled)"
        lines = [f"mode_seeking_ratio: {feedback.mode_seeking_ratio:.3f}"]
        if feedback.attribute_sensitivity:
            lines.append("per-attribute sensitivity (text distance for 1-attr-changed pairs):")
            for k, v in sorted(
                feedback.attribute_sensitivity.items(), key=lambda kv: kv[1]
            ):
                lines.append(f"  - {k}: {v:.3f}")
        return "\n".join(lines)

    def _format_banned(self, feedback: IterationFeedback) -> str:
        if not feedback.banned_phrasings:
            return "(none yet)"
        return "\n".join(f"- {p!r}" for p in feedback.banned_phrasings)

    def _format_coverage_holes(self, feedback: IterationFeedback) -> str:
        if not feedback.coverage_hole_exemplars:
            return "(none flagged)"
        return "\n".join(f"- {e}" for e in feedback.coverage_hole_exemplars[:5])

    def _format_attribute_failures(self, fails: list[AttributeVerdict]) -> str:
        if not fails:
            return "(none in this iteration)"
        bullet = []
        for f in fails[: self.max_failures]:
            bullet.append(
                f"- {f.sample_id}: failed={f.failed_attributes} reason={f.reason}"
            )
        if len(fails) > self.max_failures:
            bullet.append(f"... and {len(fails) - self.max_failures} more")
        return "\n".join(bullet)

    def _format_realism_artifacts(self, artifacts: list[RealismVerdict]) -> str:
        if not artifacts:
            return "(no detected synthetic samples this iteration)"
        bullet = []
        for v in artifacts[: self.max_artifacts]:
            bullet.append(f"- {v.sample_id} ({v.confidence:.2f}): {v.reason}")
        if len(artifacts) > self.max_artifacts:
            bullet.append(f"... and {len(artifacts) - self.max_artifacts} more")
        return "\n".join(bullet)

    def _format_diversity(self, report: DiversityReport) -> str:
        return json.dumps(
            {
                "summary": report.summary,
                "missing_modes": report.missing_modes,
                "overrepresented_modes": report.overrepresented_modes,
                "near_duplicate_rate": report.near_duplicate_rate,
                "recommendations": report.recommendations,
                "coverage": report.coverage,
            },
            indent=2,
        )
