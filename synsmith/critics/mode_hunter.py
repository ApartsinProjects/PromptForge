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

from synsmith.llm import LLMClient, json_chat
from synsmith.schema import RealExample, SyntheticSample


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
    # Generic register-anchor knob (see _is_domain_canonical below).
    # When True (default), reject candidate bans whose content words ALL
    # appear in the real corpus: these are statistically rare in the
    # small real seed but domain-canonical in the broader real
    # distribution. Without this veto, the Mode Hunter at small real-seed
    # sizes systematically pushes the generator away from the target
    # domain (e.g., banning 'the visuals are' / 'the performances are'
    # on a film-criticism corpus).
    veto_domain_canonical: bool = True


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
        real_texts = [e.text for e in real]

        for entry in obj.get("findings", []):
            pattern = str(entry.get("pattern", "")).strip()
            if not pattern:
                continue
            # Validate the LLM's claim deterministically: at least min_repeats
            # synthetic occurrences AND zero real occurrences.
            n_synth = self._count_substring(pattern, [s.text for s in synthetic])
            n_real_obs = self._count_substring(pattern, real_texts)
            if n_synth < self.config.min_repeats or n_real_obs > 0:
                continue
            # Generic register-anchor veto: refuse to ban patterns whose
            # content words all appear in the real seed. At small real-seed
            # sizes, the literal phrasing is statistically rare by chance,
            # but the underlying lexical content is domain-canonical and
            # the generator should be free to use it.
            if (
                self.config.veto_domain_canonical
                and self._is_domain_canonical(pattern, real_texts)
            ):
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

    # English stopwords that don't carry domain signal. Conservative list:
    # function words, common auxiliaries, demonstratives. Domain veto looks
    # at the remaining "content words" only.
    _STOPWORDS: frozenset[str] = frozenset(
        {
            "a", "an", "and", "are", "as", "at", "be", "been", "by", "do",
            "does", "did", "for", "from", "had", "has", "have", "he", "her",
            "him", "his", "i", "in", "is", "it", "its", "me", "my", "no",
            "not", "of", "on", "or", "our", "she", "so", "that", "the",
            "their", "them", "there", "they", "this", "to", "us", "was",
            "we", "were", "what", "when", "which", "who", "will", "with",
            "would", "you", "your",
        }
    )

    # Fraction of pattern content words that must morphologically match a
    # real-seed token before the pattern is judged domain-canonical.
    _DOMAIN_OVERLAP_THRESHOLD: float = 0.5

    @classmethod
    def _is_domain_canonical(cls, pattern: str, real_texts: list[str]) -> bool:
        """Generic check: does pattern's content vocabulary overlap real's?

        A pattern is judged DOMAIN-CANONICAL (and the ban is vetoed) when
        at least ``_DOMAIN_OVERLAP_THRESHOLD`` of its content words have a
        morphological match in the real seed. A morphological match is a
        4-character prefix in common (so ``visuals`` matches ``visual``,
        ``pacing`` matches ``pace``, ``credits`` matches ``credit``), or
        an exact equality for words shorter than 4 characters.

        Rationale: at small real-seed sizes (N=60 is typical), the literal
        phrasing of a domain-canonical n-gram is statistically likely to
        be absent from the seed by chance alone, even when the lexical
        content is domain-relevant. A strict "all content words appear
        verbatim" check has poor recall in that regime; the 50% threshold
        with 4-char-prefix stemming is a deterministic, language-agnostic
        proxy for "would this n-gram appear in the broader real
        distribution this seed was sampled from?".

        Worked examples on a 60-sample SST-2 film-critic seed:
        - ``the visuals are`` -> content=[visuals]; ``visual`` is in real;
          1/1 = 100% >= 50% -> VETO ban.
        - ``Oh sure,`` -> content=[sure]; if ``sure`` is in real, VETO.
        - ``long after the credits roll`` -> content=[long, after, credits,
          roll]; long+after match real; 2/4 = 50% -> VETO.
        - ``xyzzy plover`` -> 0/2 = 0% -> allow ban (genuine artifact).
        """
        if not pattern or not real_texts:
            return False
        pat_tokens = [
            "".join(c for c in tok if c.isalnum()).lower()
            for tok in pattern.split()
        ]
        content = [t for t in pat_tokens if t and len(t) > 2 and t not in cls._STOPWORDS]
        if not content:
            return False  # nothing to judge by; let the strict veto fire.
        # Build a real-corpus token set (alnum-only, lower).
        real_tokens: set[str] = set()
        for text in real_texts:
            for tok in text.split():
                clean = "".join(c for c in tok if c.isalnum()).lower()
                if clean:
                    real_tokens.add(clean)
        # Build a 4-char prefix set for stem-style matching.
        real_prefixes: set[str] = {t[:4] for t in real_tokens if len(t) >= 4}
        matches = 0
        for word in content:
            if word in real_tokens:
                matches += 1
                continue
            if len(word) >= 4 and word[:4] in real_prefixes:
                matches += 1
                continue
        return matches / len(content) >= cls._DOMAIN_OVERLAP_THRESHOLD

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
