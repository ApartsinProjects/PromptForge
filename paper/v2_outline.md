# v2 paper outline: target results and section structure

This file is forward-looking. It states the result targets the v2 experiments
must achieve to justify the rewrite, the section structure that would land
those results cleanly, and the decision rule for whether to merge v2 to main.

## v2 thesis (one sentence)

The seven-critic SynSmith loop produces synthetic data whose diversity
yields a measurable downstream advantage on tasks with semantically
overlapping classes, manifests as lower variance and worst-class
improvements on saturated tasks, and combines cleanly with two no-cost
inference-time modifications (test-time back-translation + quality-weighted
augmentation) that lift the augmented classifier above the real-only ceiling
on both task families.

## What v1 already established

- Lexical diversity: SynSmith has the highest distinct-n and the lowest
  self-BLEU-4 among all iterated conditions (Table 8). Significant.
- Variance reduction: on the hardest class of the customer-support task,
  SynSmith has 3.4x lower seed-variance than full_classic and is the only
  iterated condition that never collapses to F1 = 0. Robust observation.
- Protocol dependence: TF-IDF isolated gap is significant
  (-0.144, p = 0.046); the same gap is absorbed under sentence-transformer
  features. Cross-classifier robustness check.
- Post-hoc audit (Pack Discriminator below null reference, Mode Hunter tic
  count): methodological contribution that retires "n/a" ablation tables.
- MMD: SynSmith's diversity gain does not come at a measurable
  distributional cost (paired-t p in [0.30, 0.68] for all three feature
  spaces).

## What v1 failed to establish (and v2 must)

- A statistically significant aggregate F1 advantage for the seven-critic
  loop. v1 reports paired-t p >= 0.74 at every n for the augmentation
  comparison; this is reported as directional only.
- A clean cross-task replication. The customer-support 5-class task with
  the 10-item test saturates at the real-only ceiling for three of the
  five classes, leaving nowhere for the diversity claim to land.
- A method modification that converts the diversity prior into a downstream
  win.

## v2 target results (the decision rule for merging to main)

To justify merging v2 -> main the experiments must produce at least three
of the following five wins. Anything less and we keep v1 as the released
paper and document v2 as future work.

W1. Banking77 augmentation gap. On the 10-class card/payment subset at
    n_real in {10, 30, 50}, seven-critic > three-critic on macro F1 with
    paired-t p < 0.05. Expected because the classes share vocabulary, so
    the keyword-anchor advantage that lets full_classic win on the easy
    customer-support task disappears.

W2. Worst-class F1 advantage. On either task family, seven-critic >
    three-critic on worst-class F1 with paired-t p < 0.05. (Worst-class
    F1 doesn't saturate even on the easy task; v1 had directional +0.20
    advantage at p = 0.37 with N = 5 seeds.)

W3. Quality-weighted augmentation effect. Re-weighting synthetic samples
    by the attribute-verifier verdict lifts the seven-critic loop's macro
    F1 above the real-only ceiling (one-sample t against the ceiling
    rejects, p < 0.05). Already directional at N = 5 (+0.043 macro;
    one-sample p = 0.18).

W4. Surface-invariance under back-translation. |macro F1 change| under
    EN-DE-EN + EN-FR-EN paraphrase is smaller for seven-critic than for
    three-critic with paired-t p < 0.05. Already directional at N = 5
    (-0.030 macro; paired-t p = 0.10).

W5. Test-time augmentation (TTA) via back-translation does not flip the
    ranking. Specifically: TTA does not promote a non-SynSmith condition
    to the macro F1 lead position. (v1 N = 5: few_shot + TTA leads at
    0.957; this is a v2 risk we must check.)

## Proposed section structure (delta from v1)

Same sections 1-6, 8, 9, 10, 11. Section 7 (Results) is restructured:

  7.1 Augmentation on saturated 5-class task (the v1 result, repositioned
      as "saturation regime")
        - Table: macro F1 vs n with realism_only / diversity_only inline
        - Same honest framing: directional, NS at N = 5

  7.2 Augmentation on overlapping 10-class task (Banking77 subset, NEW)
        - Headline: paired-t p < 0.05 (target) at every n in {10, 30, 50}
        - Mechanism: classes share keywords, so SynSmith's diversity
          can no longer be absorbed by the embedding classifier; the
          classifier-dependent tradeoff resolves in SynSmith's favor

  7.3 Worst-class F1 (NEW)
        - Both tasks
        - SynSmith's variance-reduction signal turns into a mean
          advantage when measured on the metric that doesn't saturate

  7.4 Isolated train-on-synthetic (the v1 TF-IDF gap)
        - Kept; statistically significant cost finding
        - Now framed as "the trade we accept for the variance and
          surface-invariance wins above"

  7.5 Multi-classifier evaluation (v1 Section 7.3, repositioned)

  7.6 Augmentation multi-classifier (v1 Section 7.4)

  7.7 Direct lexical diversity (v1 Section 7.5)

  7.8 NEW: Surface-invariance under back-translation
        - Two-pivot back-translation (DE, FR)
        - |F1 change| metric: SynSmith most invariant

  7.9 NEW: Quality-weighted augmentation
        - Use the verifier verdicts as classifier sample weights
        - SynSmith exceeds the real-only ceiling

  7.10 NEW: Test-time augmentation (TTA) via back-translation
        - When you have synthetic anyway, the back-translation models are
          free at inference time; aggregate across original + paraphrases
        - Honest framing: TTA boosts everything, SynSmith slightly more

Section 8 (audit) stays.

## v2 paper writing plan

1. Wait for N=10 customer-support extension to complete (~60 min).
2. Wait for Banking77 generation to complete (~60-120 min).
3. Run scripts/v2_dashboard.py --base main_run_002 and
   scripts/banking77_augmentation_eval.py --base banking77_run_001.
4. Audit the wins against W1-W5. If >= 3 hit p < 0.05, proceed with the
   rewrite. Otherwise, document the partial results in paper/v2_results.md
   and keep v1 as the released paper.
5. Branch:
   - Decision proceed -> merge v2 -> main with a single squash commit per
     section rewrite.
   - Decision hold -> keep v2 branch as the working draft.

## What we will NOT do in v2

- Per-critic surgical ablations (4 new conditions x 5 seeds). High-value
  but expensive and orthogonal to the headline question of "does
  SynSmith win on a harder task". Defer to v3.
- Verbalized Sampling baseline implementation. Defer to v3.
- Human evaluation. Defer to v3.

## Risks

- Banking77 generation cost ~$50; if the OpenAI budget runs out
  mid-run, we lose progress for the in-flight seeds (the runner is
  per-seed atomic but mid-seed crashes lose the seed).
- The 10-item v1 test set is too small to support the surface-invariance
  finding even at N = 10. Banking77's ~440-item test should fix this.
- If Banking77 also shows full_classic = full_attrforge on macro F1,
  the v2 thesis collapses. In that case we lean on Wins W3 (quality
  weighting), W4 (surface invariance), and the v1 variance findings.
