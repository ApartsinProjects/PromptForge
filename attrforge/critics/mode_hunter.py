"""Mode Hunter: an LLM whose only job is to find a mode the generator collapsed onto.

This is the closest possible analog of "gradient through the
discriminator" in a prompt-debugging setting. The Mode Hunter outputs
concrete substrings: a phrase, an opener, a bigram, a structural tic
that appears at least N times in the synthetic batch and 0 times in the
real batch.

These substrings are *appended to a persistent banned list* across
iterations. The next generator prompt is augmented with the list:
"Do NOT use any of the following phrasings, openers, or structures:
'I understand your frustration', 'Hello team', 'Thanks for reaching out
about this issue', ..."

Unlike the per-iteration auditor, the banned list is sticky: a phrasing
that disappeared at iteration 4 must NOT silently reappear at iteration
12. This is the loop's immune memory.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, Field

from attrforge.llm import LLMClient, json_chat
from attrforge.schema import RealExample, SyntheticSample


HUNTER_SYSTEM = (
    "You hunt for telltale phrasings, openers, sentence structures, or "
    "stylistic tics that appear multiple times in a synthetic batch but do "
    "not appear in a real batch. Be specific: return concrete substrings or "
    "patterns, not abstract descriptions. Return JSON only."
)

HUNTER_USER_TEMPLATE = """Synthetic batch ({n_synth} samples):
{synth_block}

Real batch ({n_real} samples):
{real_block}

Previously banned phrasings (do NOT repeat these, find NEW ones):
{previously_banned}

Find up to {max_findings} distinct telltale patterns that:
- appear in at least {min_repeats} synthetic samples, AND
- appear in 0 real samples, AND
- are not already in the banned list above.

A pattern can be a literal phrase ("I understand your"), an opener
("Hello team"), or a structural tic ("starts with the word 'so'").

Output JSON:
{{
  "findings": [
    {{
      "pattern": "<literal substring or concise structural description>",
      "n_synthetic_occurrences": <int>,
      "rationale": "<one short sentence>"
    }},
    ...
  ]
}}
"""


@dataclass
class ModeHunterConfig:
    max_findings_per_iter: int = 5
    min_repeats: int = 2
    max_banned_total: int = 50  # hard cap to prevent prompt bloat
    temperature: float = 0.0


class ModeHunterFinding(BaseModel):
    pattern: str
    n_synthetic_occurrences: int = 0
    rationale: str = ""
    introduced_at_iteration: int = 0


class ModeHunterResult(BaseModel):
    new_findings: list[ModeHunterFinding] = Field(default_factory=list)
    banned_library: list[ModeHunterFinding] = Field(default_factory=list)


class ModeHunter:
    """Stateful: keeps a persistent banned-pattern library across iterations."""

    def __init__(
        self,
        client: LLMClient,
        config: ModeHunterConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or ModeHunterConfig()
        self._library: list[ModeHunterFinding] = []

    @property
    def library(self) -> list[ModeHunterFinding]:
        return list(self._library)

    def hunt(
        self,
        real: list[RealExample],
        synthetic: list[SyntheticSample],
        *,
        iteration: int,
    ) -> ModeHunterResult:
        if not real or not synthetic:
            return ModeHunterResult(new_findings=[], banned_library=self.library)

        previously_banned = (
            "\n".join(f"- {f.pattern!r}" for f in self._library)
            or "(none yet)"
        )
        synth_block = "\n".join(
            f"{i + 1}. {s.text[:240]}" for i, s in enumerate(synthetic[:20])
        )
        real_block = "\n".join(
            f"{i + 1}. {e.text[:240]}" for i, e in enumerate(real[:20])
        )
        user_msg = HUNTER_USER_TEMPLATE.format(
            n_synth=len(synthetic),
            n_real=len(real),
            synth_block=synth_block,
            real_block=real_block,
            previously_banned=previously_banned,
            max_findings=self.config.max_findings_per_iter,
            min_repeats=self.config.min_repeats,
        )
        try:
            obj = json_chat(
                self.client,
                HUNTER_SYSTEM,
                [{"role": "user", "content": user_msg}],
                temperature=self.config.temperature,
                max_tokens=600,
                retries=1,
            )
        except Exception:
            return ModeHunterResult(new_findings=[], banned_library=self.library)

        new_findings: list[ModeHunterFinding] = []
        seen_lower = {f.pattern.strip().lower() for f in self._library}

        for entry in obj.get("findings", []):
            pattern = str(entry.get("pattern", "")).strip()
            if not pattern:
                continue
            # Validate the LLM's claim deterministically: at least min_repeats
            # synthetic occurrences AND zero real occurrences.
            n_synth = self._count_substring(pattern, [s.text for s in synthetic])
            n_real_obs = self._count_substring(pattern, [e.text for e in real])
            if n_synth < self.config.min_repeats or n_real_obs > 0:
                continue
            key = pattern.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            new_findings.append(
                ModeHunterFinding(
                    pattern=pattern,
                    n_synthetic_occurrences=n_synth,
                    rationale=str(entry.get("rationale", "")),
                    introduced_at_iteration=iteration,
                )
            )

        # Append to the persistent library; cap at the configured budget.
        self._library.extend(new_findings)
        if len(self._library) > self.config.max_banned_total:
            self._library = self._library[-self.config.max_banned_total :]

        return ModeHunterResult(
            new_findings=new_findings,
            banned_library=self.library,
        )

    def render_for_prompt(self) -> str:
        """Render the banned library as a bullet list for the generator prompt."""
        if not self._library:
            return ""
        bullets = "\n".join(f"- {f.pattern!r}" for f in self._library)
        return (
            "Do NOT use any of the following phrasings, openers, or "
            "structural tics (they are telltale LLM signatures observed in "
            "previous batches):\n" + bullets
        )

    @staticmethod
    def _count_substring(pattern: str, corpus: list[str]) -> int:
        """Case-insensitive substring count: number of documents that contain pattern."""
        p = pattern.strip().lower()
        if not p:
            return 0
        return sum(1 for t in corpus if p in t.lower())

    @staticmethod
    def top_ngrams(texts: list[str], n: int = 3, top_k: int = 20) -> list[tuple[str, int]]:
        """Utility: deterministic n-gram histogram, useful as a fallback when no LLM is available."""
        counts: Counter = Counter()
        for t in texts:
            tokens = t.lower().split()
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                counts[gram] += 1
        return counts.most_common(top_k)
