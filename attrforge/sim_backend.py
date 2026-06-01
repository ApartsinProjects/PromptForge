"""Prompt-sensitive simulated LLM backend.

Purpose
-------
When live-API credit is unavailable, we still need to demonstrate that the
AttrForge loop *responds to its critic feedback* in a measurable way. A
trivial echo backend cannot do this: its output is independent of the
prompt, so every condition (naive, self-critique, full AttrForge) produces
identical samples and the experiment is degenerate.

This module provides a deterministic, dependency-free pseudo-LLM whose
output IS sensitive to specific prompt content. Concretely, the generator
side of the simulator:

* Picks a base utterance for the requested label.
* Applies attribute-conditioned surface mutations (style, noise,
  scenario_type).
* Applies *prompt-conditioned* mutations driven by structured phrases the
  prompt updater is encouraged to use:

  - ``forbidden phrasings: <list>``    -> never emit those substrings
  - ``include typos``                  -> add typos
  - ``include fragments``              -> drop sentence-ends
  - ``vary openers``                   -> rotate opener pool
  - ``increase ambiguity``             -> add hedge words
  - ``rare scenario``                  -> append weird-context noun
  - ``concise``                        -> truncate
  - ``verbose``                        -> append boilerplate

This is NOT a real LLM. It is an experimentally-fair *harness validator*:
all conditions are evaluated by the SAME simulator with the SAME seed.
Conditions whose prompt-update loop produces richer prompts (richer
forbidden-list, more diverse instructions) measurably outperform those
that do not.

Critic backends in simulation
-----------------------------
The discriminator and verifier need to return structured JSON. We provide
a heuristic JSON responder that:

* Verifier: rule-based attribute match using a small lexicon.
* Discriminator: heuristic real-vs-synth based on length and presence of
  forbidden openers (matches how the simulator emits text).
* Pack discriminator: classifies the pack with more LLM-like tics as the
  LLM pack.
* Auditor / Mode Hunter: returns missing modes / phrases based on a
  trivial frequency check.
* Updater: rewrites the prompt by appending instruction clauses derived
  from feedback bullets.

The behavior matters for one thing only: that conditions which receive
*structurally richer* feedback produce *structurally richer* prompts,
which produce *structurally more diverse* samples. The simulator was
designed so this gradient exists.

Reproducibility: every randomness draw uses a hash of (prompt + sample
id + seed), so the same input always gives the same output.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass

from attrforge.llm import LLMConfig

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

LABEL_BASES = {
    "refund_request": [
        "Please refund my last order, it never arrived.",
        "I was double charged. Can you reverse one of them.",
        "The item arrived damaged and I want my money back.",
        "Subscription auto-renewed after I cancelled. Refund please.",
        "I want a refund for the missing item, not a reship.",
        "Card was charged for the annual plan, I only wanted monthly.",
        "Got the wrong size. Want my money back, not a store credit.",
        "Transaction failed but the money was deducted, can you refund.",
    ],
    "technical_problem": [
        "The app keeps crashing when I open the camera tab.",
        "I get a 502 error every time I upload a file over five megabytes.",
        "The checkout button does nothing, just an endless spinner.",
        "Sync is broken between desktop and mobile.",
        "CSV import times out around three thousand rows.",
        "The export to PDF button is greyed out for me only.",
        "Search filters don't apply on Firefox latest.",
        "Mobile app won't open since the last update.",
    ],
    "account_issue": [
        "Forgot my password and the reset email never arrives.",
        "Two factor codes never come through, I'm locked out for two days.",
        "Deleted my account by mistake yesterday, can I undo that.",
        "My email got hijacked and the password was changed.",
        "Phone number on my account is wrong but the edit field is disabled.",
        "MFA app got wiped when I changed phones, please disable it.",
        "I changed my email and can't receive verification codes anymore.",
        "Can't log in even though I just reset the password.",
    ],
    "complaint": [
        "This is the third week with no resolution. I want a manager.",
        "Your support told me 24 hours, it has been four days, this is unacceptable.",
        "I have been a customer for years and this is the worst experience.",
        "Still no answer on my ticket from last Monday.",
        "Your new pricing is a joke. Why would anyone pay this.",
        "I paid for the upgrade but features are still locked.",
        "Cancel my plan. I'm done.",
        "Honestly the quality has gone downhill.",
    ],
    "general_question": [
        "Do you ship to Norway? Asking before signing up.",
        "Is the pro plan billed monthly or yearly.",
        "What is the difference between the pro and the business tier.",
        "Are there planned outages this weekend.",
        "Do you have a student discount, and how do I apply.",
        "Does the API have a rate limit on the search endpoint.",
        "Can two people share one account or do we need separate seats.",
        "Any chance of a self hosted option in the future.",
    ],
}

STYLE_OPENERS = {
    "formal": ["Hello team,", "Dear Support,", "To whom it may concern,", "Good day,"],
    "informal": ["hey", "hi", "yo", "hey there,"],
    "fragmented": ["", "uh", "ok so", "look"],
    "emotional": ["I am very upset.", "I'm fuming.", "this is exhausting,"],
    "concise": [""],
    "verbose": [
        "I hope this message finds you well. I am writing to bring to your attention an issue I have been experiencing.",
    ],
}

NOISE_TYPO_MAP = {"the": "teh", "you": "yu", "and": "annd", "are": "ar", "this": "thsi", "have": "hav", "this": "thsi"}

SCENARIO_TAILS = {
    "common": [""],
    "rare": [" Edge case: only happens on Tuesdays.", " Note: appears once a year."],
    "edge_case": [
        " Specific case: shipping address has Cyrillic characters.",
        " Note: locale set to en-XB.",
        " This only fails for accounts created before 2017.",
    ],
}

AMBIGUITY_HEDGES = {
    "low": [""],
    "medium": [" Maybe related, not sure.", " Could be a config issue, hard to tell."],
    "high": [
        " It's hard to say if this is a bug or expected.",
        " Honestly I'm not sure if it's your end or mine.",
        " Might just be me, but it's confusing.",
    ],
}

# Phrases the simulator considers "telltale LLM tics". A prompt that bans
# any of these will cause the simulator to drop them from candidate text.
DEFAULT_LLM_TICS = [
    "I understand your frustration",
    "Thanks for reaching out",
    "I appreciate your patience",
    "Please rest assured",
    "I apologize for any inconvenience",
    "We value your feedback",
]


# ---------------------------------------------------------------------------
# Sim client
# ---------------------------------------------------------------------------


def _seed_of(*parts: str) -> int:
    h = hashlib.blake2b(("||".join(parts)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")


def _choice(pool: list, seed: int):
    if not pool:
        return ""
    return pool[seed % len(pool)]


def _detect_target(prompt: str) -> dict:
    """Best-effort extraction of the target attribute vector from a generator prompt."""
    out: dict = {}
    m = re.search(r"Target attributes for this sample:\s*(\{[^}]*\})", prompt, re.DOTALL)
    if m:
        try:
            out = json.loads(m.group(1))
        except Exception:
            out = {}
    if not out:
        # Look for ``key: value`` lines in a target block.
        for line in prompt.splitlines():
            mm = re.match(r"\s*[\"']?(\w+)[\"']?\s*:\s*[\"']?([\w_]+)[\"']?", line)
            if mm:
                k, v = mm.group(1), mm.group(2)
                if k in {"intent", "label", "difficulty", "ambiguity", "style", "noise", "scenario_type"}:
                    out.setdefault(k, v)
    m = re.search(r"Sample id:\s*(\S+)", prompt)
    if m:
        out["_sample_id"] = m.group(1).strip().rstrip(",")
    return out


def _detect_forbidden(prompt: str) -> list[str]:
    """Find any banned/forbidden phrasings the updater added to the prompt."""
    forbidden: list[str] = []
    # Quoted phrases following words like "do not use" / "avoid" / "forbidden"
    keywords = r"(do not use|avoid|forbidden|never use|don't use|do NOT use)"
    for m in re.finditer(rf"{keywords}[^\n]*?['\"“]([^'\"”]{{3,}})['\"”]", prompt, re.IGNORECASE):
        forbidden.append(m.group(2))
    # Bullet lists under headings that name banned/forbidden phrasings.
    sections = re.split(r"\n\s*\n", prompt)
    for sec in sections:
        if re.search(r"(banned|forbidden)\s+phras", sec, re.IGNORECASE):
            for line in sec.splitlines():
                m = re.match(r"\s*[-*]\s+['\"“]([^'\"”]{3,})['\"”]", line)
                if m:
                    forbidden.append(m.group(1))
    return forbidden


def _detect_instructions(prompt: str) -> set[str]:
    """Detect a small vocabulary of structural instructions the updater can issue."""
    p = prompt.lower()
    tags: set[str] = set()
    if "include typo" in p or "typos" in p:
        tags.add("typos")
    if "fragment" in p or "incomplete" in p:
        tags.add("fragments")
    if "vary opener" in p or "vary the opener" in p or "different opener" in p or "rotate opener" in p:
        tags.add("vary_openers")
    if "ambig" in p:
        tags.add("ambiguity")
    if "rare" in p or "edge case" in p:
        tags.add("rare")
    if "diverse" in p or "diversity" in p:
        tags.add("diverse")
    if "concise" in p:
        tags.add("concise")
    if "verbose" in p:
        tags.add("verbose")
    if "noise" in p:
        tags.add("noise")
    return tags


def _apply_typos(s: str, seed: int) -> str:
    words = s.split()
    if not words:
        return s
    rng = seed
    n_typos = max(1, len(words) // 8)
    for _ in range(n_typos):
        idx = rng % len(words)
        rng = rng * 1664525 + 1013904223
        w = words[idx].lower().strip(string.punctuation)
        if w in NOISE_TYPO_MAP:
            words[idx] = words[idx].replace(w, NOISE_TYPO_MAP[w])
        elif len(w) > 4:
            # transpose two adjacent letters
            pos = (rng % (len(w) - 2)) + 1
            w2 = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2 :]
            words[idx] = words[idx].replace(w, w2)
    return " ".join(words)


def _maybe_insert_tic(
    text: str, available_tics: list[str], seed: int, suppression: float = 0.0
) -> str:
    """When tics are not forbidden, the simulator likes to add one.

    ``suppression`` in [0, 1] reduces the base 35% emission rate. The prompt
    updater earns suppression by listing more forbidden phrasings or by
    explicitly instructing variation.
    """
    if not available_tics:
        return text
    tic = available_tics[seed % len(available_tics)]
    base_rate = 35
    rate = max(0, int(base_rate * (1.0 - suppression)))
    if seed % 100 < rate:
        return f"{tic}, {text[0].lower()}{text[1:]}" if text else tic
    return text


def _generate_text(prompt: str, target: dict, sample_id: str, run_seed: int) -> str:
    label = target.get("intent") or target.get("label") or "general_question"
    style = target.get("style", "informal")
    noise = target.get("noise", "clean")
    scenario = target.get("scenario_type", "common")
    ambiguity = target.get("ambiguity", "low")
    difficulty = target.get("difficulty", "medium")

    seed = _seed_of(str(run_seed), sample_id, label, style, noise, scenario)
    base_pool = LABEL_BASES.get(label, LABEL_BASES["general_question"])
    base = _choice(base_pool, seed >> 3)

    instructions = _detect_instructions(prompt)
    forbidden = _detect_forbidden(prompt)
    forbidden_lower = [f.lower() for f in forbidden]

    available_tics = [t for t in DEFAULT_LLM_TICS if t.lower() not in forbidden_lower]

    parts: list[str] = []

    # Opener pool selection. "vary_openers" instruction rotates across the
    # pool; absence means deterministic selection (mode collapse on opener).
    opener_pool = STYLE_OPENERS.get(style, [""])
    if "vary_openers" in instructions or "diverse" in instructions:
        opener = _choice(opener_pool, seed >> 5)
    else:
        opener = opener_pool[0]
    # Formal openers are the largest single discriminator cue. The updater
    # can replace them with informal greetings by including the explicit
    # instruction. This is what closes the realism gap over iterations.
    if style == "formal" and "vary_openers" in instructions and "informal" in str(instructions):
        opener = ""
    if opener:
        parts.append(opener)

    body = base
    # Suppression rises with the size of the forbidden list and with the
    # presence of the explicit "vary openers" instruction. A loop that has
    # accumulated banned phrasings drives tic insertion to zero.
    suppression = 0.0
    if forbidden:
        suppression += min(1.0, 0.25 * len(forbidden))
    if "vary_openers" in instructions:
        suppression += 0.3
    suppression = min(1.0, suppression)
    body = _maybe_insert_tic(body, available_tics, seed >> 7, suppression=suppression)

    # Apply noise
    if noise == "typos" or "typos" in instructions:
        body = _apply_typos(body, seed >> 9)
    if noise == "missing_details" or "fragments" in instructions:
        # Drop the second sentence or trailing clause.
        body = body.split(".")[0].strip() + "."
    if noise == "irrelevant_details":
        body = body + " By the way the weather is bad."
    if noise == "contradiction":
        body = body + " Actually it might be fine."

    # Ambiguity
    if "ambiguity" in instructions or ambiguity in ("medium", "high"):
        body = body + _choice(AMBIGUITY_HEDGES.get(ambiguity, [""]), seed >> 11)

    # Scenario tail
    if "rare" in instructions or scenario in ("rare", "edge_case"):
        body = body + _choice(SCENARIO_TAILS.get(scenario, [""]), seed >> 13)

    # Style modifiers
    if style == "concise" or "concise" in instructions:
        body = body.split(".")[0].strip()
    if style == "verbose" or "verbose" in instructions:
        body = body + " Please advise at your earliest convenience."

    text = (" ".join(parts) + " " + body).strip()

    # Final forbidden phrase rinse
    for f in forbidden:
        text = re.sub(re.escape(f), "", text, flags=re.IGNORECASE).strip()

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _verifier_response(prompt: str, run_seed: int) -> str:
    """Heuristic verifier: match on label keyword presence; difficulty 'hard' fails when keyword is obvious."""
    target = _detect_target(prompt)
    sample_id = target.get("_sample_id", "S")
    # Extract the text under quotes.
    m = re.search(r'text:\s*"""(.*?)"""', prompt, re.DOTALL)
    text = (m.group(1) if m else "").lower()
    label = target.get("intent", target.get("label", ""))

    keywords = {
        "refund_request": ["refund", "money back", "charge"],
        "technical_problem": ["crash", "error", "broken", "doesn't work", "won't", "timeout", "502"],
        "account_issue": ["password", "login", "account", "verification", "mfa", "locked out"],
        "complaint": ["unacceptable", "manager", "ridiculous", "worst", "fuming"],
        "general_question": ["?", "do you", "is the", "what is", "any chance"],
    }
    kws = keywords.get(label, [])
    keyword_hits = sum(1 for k in kws if k in text)
    label_match = keyword_hits > 0 or not kws

    difficulty = target.get("difficulty", "medium")
    if difficulty == "hard" and keyword_hits >= 1 and len(text) < 80:
        # Hard examples should not be too obvious.
        label_match = False
        failed = ["difficulty"]
        reason = "label keyword is too prominent for a hard example."
    elif not label_match:
        failed = ["intent" if "intent" in target else "label"]
        reason = "text does not clearly reflect the requested label."
    else:
        failed = []
        reason = "ok"

    return json.dumps(
        {
            "sample_id": sample_id,
            "attribute_match": len(failed) == 0,
            "failed_attributes": failed,
            "reason": reason,
        }
    )


def _discriminator_response(prompt: str, run_seed: int) -> str:
    """Heuristic discriminator: detects tics + within-batch opener uniformity.

    Two cues are used:
    1. Presence of known LLM tic substrings.
    2. The sample's opener is the modal opener in the synthetic batch
       (a within-batch homogeneity signal).

    A sample is flagged synthetic if it triggers either cue. As the loop
    suppresses tics (via banned-phrasings) and rotates openers (via the
    vary-openers instruction), this discriminator's accuracy on a balanced
    real/synth mix drifts toward chance, which is the realism objective.
    """
    samples = re.findall(
        r"\[sample_id:\s*([^\]]+)\]\s*(.+?)(?=\n\[sample_id:|\Z)", prompt, re.DOTALL
    )
    # Compute the modal opener across the batch (first 4 tokens of each text).
    from collections import Counter

    openers = []
    for _, text in samples:
        toks = text.strip().split()[:4]
        if toks:
            openers.append(" ".join(toks).lower())
    opener_count = Counter(openers)
    modal_opener, modal_n = opener_count.most_common(1)[0] if opener_count else ("", 0)
    # Modal opener counts as a homogeneity cue only if used by >=30% of batch.
    modal_share = modal_n / max(1, len(openers))
    homogeneity_signal = modal_share >= 0.30

    out = []
    for sid, text in samples:
        sid = sid.strip()
        t = text.lower()
        tic_hits = sum(1 for tic in DEFAULT_LLM_TICS if tic.lower() in t)
        sample_opener = " ".join(t.strip().split()[:4])
        opener_match = (
            homogeneity_signal and sample_opener == modal_opener and modal_share >= 0.30
        )
        is_synth = tic_hits > 0 or opener_match
        out.append(
            {
                "sample_id": sid,
                "prediction": "synthetic" if is_synth else "real",
                "confidence": 0.85 if tic_hits > 0 else 0.65 if opener_match else 0.55,
                "reason": (
                    "contains a known LLM tic phrase."
                    if tic_hits > 0
                    else f"shares opener with {int(modal_share * 100)}% of the batch."
                    if opener_match
                    else "natural length and tone."
                ),
            }
        )
    return json.dumps(out)


def _auditor_response(prompt: str, run_seed: int) -> str:
    """Heuristic auditor based on coverage block."""
    coverage_lines = re.findall(r"\s*(\w+):\s*([0-9.]+)", prompt)
    missing = []
    overrep = []
    for name, frac in coverage_lines:
        f = float(frac)
        if 0.0 <= f <= 1.0 and f < 0.8 and name not in {"near_duplicate_rate"}:
            missing.append(f"more variety in attribute '{name}' (only {int(f * 100)}% of values seen)")
    if not missing:
        missing = ["rare scenarios with missing details"]
    return json.dumps(
        {
            "summary": "deterministic coverage observation",
            "missing_modes": missing[:4],
            "overrepresented_modes": ["clean medium-length examples"],
            "near_duplicate_rate": 0.1,
            "recommendations": [
                "Force generation across rare scenario types.",
                "Vary openers and structure.",
            ],
        }
    )


def _pack_response(prompt: str, run_seed: int) -> str:
    """Heuristic pack discriminator: counts shared openers / phrases in each pack."""

    def telltale_score(block: str) -> int:
        b = block.lower()
        score = 0
        for o in STYLE_OPENERS["formal"]:
            score += b.count(o.lower()) >= 2
        for tic in DEFAULT_LLM_TICS:
            score += b.count(tic.lower()) >= 2
        return int(score)

    m_a = re.search(r"Pack A.*?:\s*(.+?)Pack B", prompt, re.DOTALL)
    m_b = re.search(r"Pack B.*?:\s*(.+)", prompt, re.DOTALL)
    a_block = m_a.group(1) if m_a else ""
    b_block = m_b.group(1) if m_b else ""
    a_score = telltale_score(a_block)
    b_score = telltale_score(b_block)
    if a_score == b_score:
        chosen = "A" if (run_seed % 2 == 0) else "B"
    else:
        chosen = "A" if a_score > b_score else "B"
    return json.dumps(
        {
            "llm_pack": chosen,
            "confidence": 0.6 + 0.1 * abs(a_score - b_score),
            "shared_patterns": (
                ["uniform formal opener across pack"] if max(a_score, b_score) > 0 else []
            ),
        }
    )


def _hunter_response(prompt: str, run_seed: int) -> str:
    """Look for an active LLM tic that appears in the synthetic block more than once."""
    m = re.search(r"Synthetic batch.*?:\s*(.+?)Real batch", prompt, re.DOTALL)
    synth = (m.group(1) or "").lower() if m else ""
    m_b = re.search(r"banned\s*\(.*?\):\s*(.*?)Find", prompt, re.DOTALL | re.IGNORECASE)
    banned = (m_b.group(1) or "").lower() if m_b else ""
    findings = []
    for tic in DEFAULT_LLM_TICS:
        if tic.lower() in banned:
            continue
        c = synth.count(tic.lower())
        if c >= 2:
            findings.append(
                {
                    "pattern": tic,
                    "n_synthetic_occurrences": c,
                    "rationale": "boilerplate LLM phrase repeated across samples.",
                }
            )
        if len(findings) >= 2:
            break
    return json.dumps({"findings": findings})


_STYLE_HEADER = "Style and constraints (added by updater):"


def _updater_response(prompt: str, run_seed: int) -> str:
    """Rewrite the prompt with deduplicated instructions and forbidden phrasings.

    The previous implementation appended a fresh block every iteration, which
    duplicated text when the same critic signals fired across rounds. We now
    parse the existing block, merge new clauses into it, drop duplicates, and
    cap the result at MAX_CLAUSES to enforce a soft length budget.
    """
    m = re.search(r'Current generator prompt:\s*"""(.*?)"""', prompt, re.DOTALL)
    current = (m.group(1) or "").strip() if m else ""

    # Split the previous prompt into body + existing instruction block.
    body, _, existing_block = current.partition(_STYLE_HEADER)
    body = body.rstrip()
    existing_clauses = [
        re.sub(r"^-\s+", "", line).strip()
        for line in existing_block.splitlines()
        if line.strip().startswith("-")
    ]

    new_clauses: list[str] = []

    if re.search(r"(synthetic|realism)", prompt, re.IGNORECASE) and re.findall(
        r"^\s*-\s*\S+\s*\([^)]*\):\s*(.+)$", prompt, re.MULTILINE
    ):
        new_clauses.append(
            "Vary openers and avoid template phrasings such as boilerplate apologies."
        )

    banned_section = re.search(
        r"Persistent banned phrasings library.*?:\s*(.+?)(?:\n\n|\nReal exemplars)",
        prompt,
        re.DOTALL,
    )
    if banned_section:
        banned_items = re.findall(
            r"-\s+['\"“]([^'\"”]+)['\"”]", banned_section.group(1)
        )
        if banned_items:
            quoted = ", ".join(f'"{b}"' for b in banned_items[:8])
            new_clauses.append(f"Forbidden phrasings (do NOT use): {quoted}.")

    if "missing_modes" in prompt:
        new_clauses.append(
            "Include rare and edge_case scenario types, fragmented and emotional "
            "styles, and varied noise levels including typos and missing_details."
        )

    if re.search(r"pack_accuracy", prompt):
        new_clauses.append(
            "Vary structure across samples, not only within each sample."
        )

    if "mode_seeking_ratio" in prompt:
        new_clauses.append(
            "Each requested attribute change must produce a visible surface change "
            "in the text. Rotate openers per sample."
        )

    if re.search(r"Real exemplars[^:]*:", prompt):
        new_clauses.append(
            "Match the style and surface form of any provided real exemplars."
        )

    # Merge old and new, deduplicate by lowercased prefix (60 chars).
    merged: list[str] = []
    seen_prefixes: set[str] = set()
    for clause in existing_clauses + new_clauses:
        key = clause.lower().strip().rstrip(".")[:60]
        if not key or key in seen_prefixes:
            continue
        seen_prefixes.add(key)
        merged.append(clause)

    # Cap the clause list. Most recent N entries win, so banned-phrasings
    # always make it through.
    MAX_CLAUSES = 8
    if len(merged) > MAX_CLAUSES:
        merged = merged[-MAX_CLAUSES:]

    if not merged:
        return body

    block = _STYLE_HEADER + "\n" + "\n".join(f"- {c}" for c in merged)
    return (body + "\n\n" + block).strip()


class SimClient:
    """Drop-in replacement for an LLMClient that uses heuristic responses."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        # ``cfg.extra`` may carry ``role`` and ``seed``.
        self.role = (cfg.extra or {}).get("role", "generator")
        self.seed = int((cfg.extra or {}).get("seed", 17))
        self._counter = 0

    def chat(
        self,
        system: str,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._counter += 1
        user = messages[-1]["content"] if messages else ""
        combined = system + "\n\n" + user

        # Route by role hint or by content sniffing.
        if "Sample id:" in user or "Target attributes for this sample" in user:
            target = _detect_target(combined)
            sid = target.get("_sample_id") or f"sim_{self._counter:04d}"
            text = _generate_text(combined, target, sid, self.seed)
            return json.dumps(
                {"sample_id": sid, "text": text, "attributes": {k: v for k, v in target.items() if not k.startswith("_")}}
            )

        if 'requested attributes' in user and 'text:' in user:
            return _verifier_response(combined, self.seed)

        if "Pack A" in user and "Pack B" in user:
            return _pack_response(combined, self.seed)

        if "Synthetic batch" in user and "Real batch" in user and "telltale" in (system + user):
            return _hunter_response(combined, self.seed)

        if "shuffled mix" in system or "forensic reader" in system:
            return _discriminator_response(combined, self.seed)

        if "Coverage so far" in user or "Observed batch" in user:
            return _auditor_response(combined, self.seed)

        if "improving" in system.lower() or "Current generator prompt" in user:
            return _updater_response(combined, self.seed)

        return "[sim] no specific route; empty body."


def build_sim_client(cfg: LLMConfig) -> SimClient:
    return SimClient(cfg)
