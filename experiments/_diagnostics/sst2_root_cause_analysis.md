# SST-2 root-cause analysis: GENERIC critic-loop register-anchor gap

Date: 2026-06-02
Method: per-test-item inspection + per-iteration critic-output forensics.

This is a FRAMEWORK-level diagnosis. PromptForge is a generic synthetic-data
loop; the critics are expected to detect failure modes automatically and the
Updater is expected to rewrite the generator prompt accordingly. The fix
therefore lives in the critics/updater/templates, NOT in any dataset-
specific config.

## Setup

- AttrForge synth-only classifier (seed 17, 48 synth samples, sentence-transformer + LR): 75.9% acc on full SST-2 validation (872 items).
- Real-only classifier (60 seed examples, same encoder + LR): 70.4% acc.
- AttrForge BEATS real-only by +5.5pp; the gap to published DistilBERT-full (91.3%) is 15.4pp.

## The cost gap, at the datapoint level

96 test items where AttrForge fails but real-only succeeds. Random sample:
- "portentous and pretentious, the weight of water is appropriately titled" (true=neg, AF=pos)
- "by-the-numbers patient/doctor pic that covers all the usual ground" (true=neg, AF=pos)
- "stealing harvard aspires to comedic grand larceny but stands convicted of nothing more than petty theft" (true=neg, AF=pos)
- "teen movies have really hit the skids" (true=neg, AF=pos)
- "visually rather stunning, but ultimately a handsome-looking bore" (true=neg, AF=pos)
- "is an arthritic attempt at directing by callie khouri" (true=neg, AF=pos)

The pattern: **sophisticated professional film-critic register** (formal vocabulary, concessive structures, idiomatic metaphors, ironic praise). The AttrForge synthetic batch is in **colloquial review register** (direct polarity words, casual viewer voice, "I really enjoyed").

## Three GENERIC framework bugs

### Bug 1: Mode Hunter has no statistical anchoring at small real-seed sizes

Looking at the actual `mode_hunter.json` outputs from the SST-2 seed-17 run, the persistent banned library accumulated these patterns over 3 iterations:

| Banned pattern | Mode Hunter rationale | Reality |
|---|---|---|
| `'Oh sure,'` | "introduces sarcastic comments" | legitimate LLM tic |
| `"the film's pacing is"` | "appears multiple times in synth, not in real" | domain-canonical film-critic phrasing |
| `'the visuals are'` | "appears multiple times in synth, not in real" | domain-canonical |
| `'the performances are'` | "appears multiple times in synth, not in real" | domain-canonical |
| `'long after the credits roll'` | "appears multiple times in synth, not in real" | domain-canonical critic idiom |
| `'drags on'` | "appears multiple times in synth, not in real" | borderline; arguably critic-register |

The check is `n_real_obs > 0 -> reject ban`. But at N=60 real seed with 296 unique tokens (verified empirically), ANY 4-word phrasing is statistically likely to be absent by chance, even if it's the most common phrasing in the broader real distribution. The Mode Hunter's veto has near-zero statistical power at this seed size, and the generator drifts to colloquial alternatives.

### Bug 2: Coverage Hole Finder works correctly, but the Updater under-weights its output

The Coverage Hole Finder correctly surfaces real-distribution exemplars the synth is missing:
- `"is enormously good fun"` (positive, p_real=0.68)
- `"and intermittently hilarious"` (positive, p_real=0.68)
- `"formulaic and forgettable"` (negative, p_real=0.68)
- `"looking down at your watch and"` (negative, p_real=0.67)

These are EXACTLY the film-critic phrasings the synth is missing. But the Updater template treats them as "stylistic anchors" (passive suggestion), not as a MUST-INCLUDE constraint. The Updater LLM does not lift them into the next prompt as preferred phrasings; it consumes only the Mode Hunter ban-list.

### Bug 3: Realism Discriminator judges samples in isolation, not vs the real distribution

Sample verdict from iter_002: "Casual tone and relatable expression of boredom suggest a genuine reaction." → marked REAL with 0.85 confidence.

The discriminator is asking "could a human have written this?" rather than "is this drawn from the same distribution as the real seed?". Casual register always passes; register mismatch never gets caught. On SST-2 (where real is professional critic prose), the dominant failure mode is invisible to the discriminator.

## Three GENERIC framework fixes

### Fix 1: `ModeHunter._is_domain_canonical` (deterministic register-anchor veto)

Before flagging a pattern, check whether at least 50% of its content words have a morphological match (4-char prefix, handles `visuals/visual`, `films/film`, `credits/credit`) in the real seed. If yes, the pattern is built from domain-canonical vocabulary and the ban is vetoed even when the literal phrasing is absent. Controlled by `ModeHunterConfig.veto_domain_canonical=True` (new default).

Verified by unit tests (`tests/test_critics/test_mode_hunter.py::test_is_domain_canonical_*`).

### Fix 2: `UPDATER_USER_TEMPLATE` (promote `coverage_hole_block` to MUST-INCLUDE)

Template rewrite: the coverage-hole exemplars are now framed as "PREFERRED phrasings the generator MUST steer toward in the next batch", and an explicit Constraint says the next prompt MUST add a "Preferred phrasings" block listing those exemplars. The Updater LLM now lifts them positively, instead of treating the block as decorative context.

### Fix 3: `DISCRIMINATOR_SYSTEM` + `DISCRIMINATOR_USER_TEMPLATE` (distribution-anchoring)

Rewrite of the realism system prompt: explicit 3-step procedure — (1) identify the real distribution's register/vocabulary by scanning all samples; (2) classify each sample against THAT identified register, not against generic plausibility; (3) name the register mismatch as a cue. A casually-written text is now flagged as synthetic when the real seed is formal critic prose.

## Smoke-test confirmation (small)

Ran the updated framework on SST-2 seed 17 (8 samples × 2 iters, eval on 32 items):

**Fix 1 confirmed**: banned library shrunk from 5+ to 2 entries; the surviving bans are legitimate LLM tics (`'Oh sure,'`, `"who doesn't love"`); all four domain-canonical phrasings vetoed.

**Fix 2 confirmed**: iter_001 prompt now contains a `**Preferred phrasings:**` block listing Coverage Hole exemplars (`a stirring, funny`, `and intermittently hilarious`, `smartly`, etc.). The Updater is now using the positive signal.

**Fix 3 confirmed**: realism verdicts now name register mismatch as the cue ("casual phrasing; real samples are more formal and critical", "formal critique with structured analysis; aligns with real samples"). The discriminator is now distribution-aware.

**Sample quality lifted visibly** (eyeballed, 16 samples from iter_001):
- "The plot meanders aimlessly, like a lost dog at a park, making it hard to care about the characters or their fates" (metaphor; critic register)
- "Despite its ambitious premise, the film ultimately falls short, failing to engage the audience beyond mere indifference" (concessive; critic register)
- "the pacing is so agonizingly slow that you could grow a beard waiting for it to pick up" (hyperbolic critic register)

vs the OLD framework's "the movie just drags on and on, i couldn't stand the pacing" / "Oh sure, this movie was definitely a masterpiece... if you enjoy watching paint dry".

## Why the fixes are generic, not SST-2 specific

All three fixes are framework-level: they apply unchanged to ANY dataset whose real seed is small (<= a few hundred examples) and domain-specific in register (any vertical: medical notes, legal contracts, financial reports, scientific abstracts, customer-support transcripts, etc.). The principle is **register-anchoring**: every adversary needs a connection to the real distribution to avoid pushing the generator off-domain. The original framework had this connection only weakly (Coverage Hole only) and the Updater was not consuming it.

These fixes are expected to lift Banking77, TREC, customer-support, MNLI, and any future task family's synth-only accuracy without per-dataset prompt engineering. The full-scale run on SST-2 seed 17 is in progress (run-id `sst2_generic_fix_full`) to quantify the lift; cross-task validation follows.

## Why this rule of inspection generalizes

This is the canonical pattern the global rule (`root-cause via specific failing datapoints`, CLAUDE.md) is meant to surface. The aggregate ("AttrForge is +5.5pp over real-only but 15.4pp below DistilBERT-full") would never have surfaced this. Inspecting 15 specific failing items + the per-iteration critic outputs (mode_hunter.json, coverage_holes.json, realism_verdicts.jsonl) made the three bugs concrete and the fix scope precise.
