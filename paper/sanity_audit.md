# Sanity Audit of Empirical Results (Live-LLM run + simulator run)

Performed against `experiments/main_run_002_*` (live `gpt-4o-mini`, 3 seeds, 7 conditions)
and `experiments/sim_run_002_*` (deterministic simulator, 3 seeds, 7 conditions).

## Live-LLM run (main_run_002): findings

### Bug B (simulator-only): RESOLVED in live LLM
For seed 17, every condition produced distinct synthetic samples; no overlap.
The simulator's bug B (identical samples across conditions because the generator
is deterministic-by-target and the prompt-content sensitivity is too weak)
does not appear with live `gpt-4o-mini`: temperature 0.9 sampling produces
unique outputs even when prompt content is similar.

### Per-iteration F1 trajectories: NON-MONOTONIC and OFTEN DECREASING

For all seeds and most iterated conditions, F1 does not improve monotonically
through iterations. Many conditions show F1 DECREASING through iterations:

| seed | condition | iter-0 F1 | iter-1 F1 | iter-2 F1 |
|------|-----------|-----------|-----------|-----------|
| 17 | self_critique | 0.394 | 0.250 | 0.257 |
| 17 | realism_only | 0.189 | 0.300 | 0.214 |
| 17 | diversity_only | 0.200 | 0.050 | **0.627** |
| 17 | full_classic | 0.367 | 0.400 | 0.294 |
| 17 | full_attrforge | 0.313 | 0.213 | 0.367 |
| 23 | self_critique | 0.433 | 0.206 | 0.548 |
| 23 | full_classic | 0.080 | 0.067 | 0.380 |
| 23 | full_attrforge | 0.117 | 0.280 | 0.567 |
| 41 | self_critique | 0.233 | 0.114 | 0.210 |
| 41 | full_classic | 0.300 | 0.293 | 0.222 |
| 41 | full_attrforge | 0.293 | 0.249 | 0.400 |

**Observation**: iteration is not monotone in any condition. The "iter-1 dip then
recovery" pattern from the simulator does not hold in live LLM. Some seeds
recover (seed 17 diversity_only; seed 23 self_critique, full_attrforge), others
degrade (seed 41 self_critique).

### Cumulative F1 plateau finding

| condition | mean F1 ± std (cumulative) |
|-----------|----------------------------|
| naive | 0.37 ± 0.05 |
| few_shot | 0.45 ± 0.19 |
| self_critique | 0.41 ± 0.09 |
| realism_only | 0.33 ± 0.03 |
| diversity_only | 0.35 ± 0.11 |
| **full_classic** | **0.58 ± 0.05** |
| full_attrforge | 0.40 ± 0.06 |

**Big surprise**: `full_classic` (3 critics) DOMINATES every other condition,
including `full_attrforge` (7 critics) by 0.18 absolute F1. Adding the four
GAN-style adversaries does NOT improve downstream F1; it appears to hurt.

### Realism discriminator drifts AWAY from chance

| condition | discriminator accuracy (chance = 0.5) |
|-----------|---------------------------------------|
| realism_only | 0.75 ± 0.43 |
| full_classic | 0.92 ± 0.14 |
| full_attrforge | 0.83 ± 0.17 |

**Big surprise #2**: the realism objective MOVES AWAY from chance through
iteration. The discriminator becomes MORE confident that samples are synthetic
as the loop runs. Adding more critics makes this worse, not better.

### Attribute fidelity is shockingly low

| condition | attribute_match_rate |
|-----------|---------------------|
| full_classic | 0.10 ± 0.10 |
| full_attrforge | 0.06 ± 0.06 |

The verifier judges that <10% of samples match their requested attribute
vector. This is well below the threshold for "attribute-controlled".

## Interpretation: a real tradeoff revealed by live LLM

The combined evidence supports a single hypothesis:

> **Adversarial prompt-updates inject surface diversity at the cost of
> downstream-classifier discriminability and at the cost of attribute fidelity.**

Concretely:
- Mode Hunter accumulates ~12 banned phrasings across 3 iterations for
  `full_attrforge` (real signal: the generator has many LLM tics worth
  suppressing).
- Pack accuracy in `full_attrforge` is 0.58 (above chance), suggesting the
  pack discriminator still detects some homogeneity, but less than naive.
- BUT the cumulative effect of these constraints is that the generator
  produces text that is HARDER to classify by intent (TF-IDF classifier
  relies on keyword consistency that the constraints disrupt).
- AND the additional verbal complexity from accumulated constraints makes
  the discriminator MORE confident in detecting synthetic patterns.

This is an honest, important finding. The paper should reposition: instead of
claiming AttrForge "improves over baselines", we report a **diagnostic
tradeoff** — AttrForge reveals what naive ablations hide.

## Bug audit (live-LLM run)

| Bug from previous audit | Status |
|-------------------------|--------|
| A. Audit RNG state evolves between conditions | FIXED in scripts/posthoc_audit.py (re-instantiates per condition) |
| B. Conditions produce identical samples | RESOLVED in live LLM (was simulator-specific) |
| C. Mode Hunter signal sparsity at batch=16 | PARTIALLY FIXED (min_repeats=1 + real LLM produces more tics) |
| D. Downstream F1 saturates regardless of condition | NOT FIXED but ACKNOWLEDGED honestly: F1 is non-monotone and the tradeoff IS the finding |
| E. Headline pack/MS claims are mostly RNG | LIKELY FIXED post-audit (running now) |

## Recommendations for paper revision

1. **Reframe contribution**: from "AttrForge improves downstream" to "AttrForge
   reveals an adversarial diversity-discriminability tradeoff invisible to
   naive ablations".
2. **Headline finding**: full_classic > full_attrforge on downstream F1; this
   IS the result, not a failure to report.
3. **Honest discussion**: more critics = more diversity = harder downstream
   classification = lower TF-IDF F1. A more capable downstream classifier
   (e.g., embedding-based) might invert this finding; we report what we measured.
4. **The realism discriminator drift** away from chance is an important
   secondary finding: the loop does NOT close the realism gap on this task;
   it widens it. Honest reporting required.
