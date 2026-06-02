# Paper-reviewer audit at v2.7

Date: 2026-06-02
Source: general-purpose subagent following paper-reviewer SKILL.md

**Recommendation: Major revision**

## Four blocking findings

**W1. The bolded headline pair in the Abstract is not uniquely attributable to `full_attrforge`.** Table 10 shows `self_critique + full_attrforge` at macro $0.947 \pm 0.056$ / worst $0.833 \pm 0.176$ and `self_critique + diversity_only` at **identical** $0.947 \pm 0.056$ / $0.833 \pm 0.176$. §7.3's leave-one-out paragraph admits this, but the Abstract / §1 / Conclusion all promote `sc + full_attrforge` as **the** headline ensemble. The honest reading: "any of two iterated conditions paired with self_critique reaches the same ceiling."

**W2. `full_attrforge` solo loses macro F1 in every Table 3 row.** At $n_{\text{real}} = 5$: `full_attrforge` 0.688 is the **worst** iterated condition (full_classic 0.759, diversity_only 0.752, realism_only 0.741, self_critique 0.699). At $n_{\text{real}} = 10$: again the worst iterated. At $n_{\text{real}} = 30$: tied with full_classic at the bottom; beaten by realism_only 0.911, self_critique 0.904, diversity_only 0.904. Table 4 per-class confirms: on `complaint` solo, full_attrforge 0.60 < full_classic 0.63 < real-only 0.67.

**W3. Banking77 shows `full_classic` ≥ `full_attrforge` on every row of every metric.** At $n_{\text{real}} = 10$: full_classic macro 0.882 > full_attrforge 0.876; worst 0.747 > 0.720. Same direction at n=30, 100, 300. Banking77 does NOT differentiate the seven-critic loop from the three-critic baseline; it slightly favors the three-critic baseline. The Abstract cites Banking77 as cross-domain confirmation without disclosing this.

**W4. The headline ensemble effect is NS against the right baseline.** Ensemble vs best individual (realism_only) is +0.036 macro F1, paired-t p=0.310, bootstrap CI [-0.027, +0.095]. The p=0.066 cited in Abstract is vs `full_attrforge` solo, which is the **worst** iterated solo (W2). Comparing the ensemble to the worst solo is not the relevant test.

## Substantive findings

**W5.** §1 contribution #4 over-claims attribution: "seven-critic loop's value is to produce decision-boundary diversity that classifier ensembling extracts". The leave-one-out finding in §7.3 says full_attrforge, realism_only, and diversity_only are individually droppable from the 5-condition ensemble "with no detectable change".

**W6.** Two of the four GAN-style adversaries (Mode-Seeking 0.23 ± 0.01 constant, Coverage AUROC 1.00 saturated) do NOT differentiate any condition. Deserves a §7 subsection or paragraph, not Figure 7 caption.

**W7.** Table 4 ensemble row uses `sc + af`; Table 10 shows it ties with `sc + diversity_only`. Either show per-class breakdown for both pairs or disclose any per-class differentiator.

**W8.** Mode Hunter library count `11.6 ± 0.6` is recoverable evidence; surface it. This is the most concrete piece of evidence that one of the four GAN-style adversaries (Mode Hunter) does work.

## Reviewer's recommendation

Three fixes restore honesty without losing the contribution:
1. Reframe the ensemble headline as "self_critique paired with any of two iterated conditions" and either drop or empirically distinguish full_attrforge.
2. State plainly in §7.1 and §7.4 that full_attrforge solo loses to full_classic (and other iterated conditions) on macro F1 on both tasks.
3. Move the NS-vs-best-solo p-value (p=0.310) into the Abstract and Conclusion alongside (or instead of) the p=0.066 vs-worst-iterated-solo number.

With these three changes the paper has a defensible TMLR-level contribution: "iteration unlocks 2× semantic diversity; ensembling two iterated conditions extracts decision-boundary diversity worth ~+0.04 macro F1 and ~1.65× variance reduction; the seven-critic loop's specific contribution is the lexical-diversity panel and the Mode Hunter banned-phrasings library, not a downstream-F1 advantage."

## What the BCa re-analysis adds

The paired BCa bootstrap + sign-flip re-analysis (scripts/paired_bca_bootstrap.py) confirms two findings the paired-t test missed:

- **H2** ensemble worst-class lift over full_attrforge solo: +0.233, BCa 95% CI **[+0.067, +0.500]** excludes zero. The +0.233 is genuinely significant under BCa.
- **H7** ensemble macro F1 over full_classic solo: +0.073, BCa 95% CI **[+0.009, +0.141]** excludes zero. The ensemble beats full_classic by ~+0.07 macro F1 with BCa CI exclusion.

And confirms the rest:
- H1 ensemble macro vs realism_only solo: +0.036, BCa [-0.033, +0.089] crosses zero. NS even under BCa.
- H4-H5 full_attrforge solo vs full_classic solo: 0.000 / -0.001. Tied.
