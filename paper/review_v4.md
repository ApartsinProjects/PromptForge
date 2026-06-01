# TMLR Fourth-Round Review (v1.4): "Adversarial Prompt Debugging for LLM Synthetic Data Generation"

Reviewer: senior reviewer
Round: 4 (post augmentation-headline pivot)
Verdict: MAJOR REVISIONS (see Section "Overall recommendation")

## Summary of changes since round 2

The authors pivoted again: the new headline is augmentation rather than isolated train-on-synth. Section 7.1 + Figure 2 + Table 3 now report sentence-transformer downstream macro F1 as a function of real-train size n in {5, 10, 15, 20, 25, 30}, with the central claim that `full_attrforge` is "the only iterated condition that reaches the real-only macro F1 ceiling (0.893) at every real-train size from 20 to 30" while `full_classic` "plateaus at 0.872". The previous TF-IDF gap is demoted to Section 7.2, the abstract is restructured as four labeled paragraphs (problem / what we do / what we find / what we additionally contribute), and a Mode-Hunter library count of 11.6 +/- 0.6 is now consistent across the paper. Run scale increased from N = 3 seeds to N = 5 seeds, which materially improves power on the TF-IDF comparison (p moved from 0.09 at N = 3 to 0.046 at N = 5) but does not save the new augmentation headline.

## What round-2 issues are now resolved

- The split-personality issue (BL-NEW-1 of round 3): Sections 9, 10, and 11 are now consistent with the abstract. No more "Live-LLM results pending" or "simulator-validated experiments". Resolved.
- The "reversal" framing (BL-NEW-2 of round 3): Section 7.3 H3 ("Isolated protocol with stronger classifiers") and the associated figure caption no longer assert a reversal; the section explicitly says "statistically indistinguishable from zero" with the p = 0.82 CI. Resolved.
- The 0.69 vs 0.79 stale per-iteration number (BL-NEW-4 of round 3): Section 7.5 now reports the correct per-iter values from per_iter.csv (0.25 +/- 0.09, 0.37 +/- 0.16, 0.49 +/- 0.19 for `full_attrforge`; 0.31 -> 0.34 -> 0.34 for `full_classic`). Verified against `experiments/main_run_002_aggregated/per_iter.csv`. Resolved.
- Section 9.1 stale numbers (BL-NEW-3 of round 3): the 0.69 / 0.72 simulator-era numbers are gone; Section 9.1 now uses 0.93 / 0.90 from main_run_002 and the paired statistics. Resolved.
- Direct diversity measurement (BL4 / Top-5 fix #5 from round 3): a new Section 7.4 with Table 6 reports distinct-1, distinct-2, distinct-3, and self-BLEU-4 across all conditions; numbers verified against `experiments/main_run_002_aggregated/diversity_metrics.json`. `full_attrforge` is highest on distinct-n and lowest on self-BLEU-4 among iterated conditions, supporting the surface-diversity claim. Resolved.
- N = 3 -> N = 5 budget concern: the new run has 5 seeds (17, 23, 41, 53, 89). The TF-IDF gap is now significant under paired-t at conventional alpha = 0.05 (p = 0.046, not p = 0.09). Resolved partially (Wilcoxon is still not in the abstract; see MINOR below).
- BibTeX and title metadata: index.html `<title>` and BibTeX entry now read "Adversarial Prompt Debugging for LLM Synthetic Data Generation" matching H1. Resolved.

## What round-2 issues remain or were re-introduced

- The "3 backends" off-by-one (round-1 M10) appears to be resolved in Section 6.2: it now reads "Four backends are implemented (`openai`, `anthropic`, `echo` for unit tests, and a deterministic prompt-sensitive `sim`)". OK.
- Section 4.4 "core conceptual claim of our architecture: structured named complaints admit locally targeted rewrites" without a controlled scalarized-vs-structured experiment (round-3 M-NEW-8) is still unfixed. The claim is still asserted; no ablation runs the same conditions with a scalarized reward.
- The Coverage Hole AUROC = 1.00 artifact-vs-real-distance ambiguity (round-3 M-NEW-6) remains. Section 8 caption says "Coverage AUROC (saturates at 1.00 for every condition)... reported in the released artifacts but does not differentiate conditions on this task". The shuffle-label or k-fold check that would distinguish "real coverage failure" from "small-N logistic-regression artifact" is still not run. The text now acknowledges the omission cleanly, but the ambiguity remains.

## New issues introduced in v1.4

### BLOCKERS

**BL-NEW-1. The new headline claim that `full_attrforge` is "the only iterated condition" that reaches the real-only ceiling at every n in [20, 30] is verifiably FALSE.**

WHERE: Abstract, "What we find" paragraph, sentence 1: "the seven-critic loop is the only iterated condition that reaches the real-only macro F1 ceiling ($0.893$) at every real-train size from $20$ to $30$". Also: Section 7.1 paragraph 2, "(ii) AttrForge reaches the real-only ceiling; full_classic does not. ... The seven-critic loop matches that ceiling at every $n \geq 20$ ($0.893 \pm 0.075$); the three-critic baseline plateaus at $0.872$ and stays there."

WHY: I verified directly from `experiments/main_run_002_aggregated/scarce_real.json`. Per-condition means under sentence-transformer + LR augmentation:

| n | real-only | self_crit | realism_only | diversity_only | full_classic | full_attrforge |
|---|-----------|-----------|--------------|----------------|--------------|----------------|
| 20 | 0.893 | 0.780 | **0.904** | 0.864 | 0.872 | 0.893 |
| 25 | 0.859 | 0.815 | **0.904** | 0.883 | 0.872 | 0.893 |
| 30 | 0.893 | 0.851 | **0.925** | **0.904** | 0.872 | 0.893 |

`realism_only` is also an iterated condition (Table 2 row labeled "realism_only" has Iters = 3). It EXCEEDS the 0.893 ceiling at all three of n = 20, 25, 30 (means 0.904, 0.904, 0.925). `diversity_only` also exceeds the 0.893 ceiling at n = 30 (0.904). Hence `full_attrforge` is NOT the only iterated condition that reaches the ceiling; in fact, by mean, `realism_only` exceeds it at every n in [20, 30] by a larger margin than `full_attrforge`.

The paired-t p-value of `full_attrforge` vs `realism_only` at these sizes is 0.81, 0.81, 0.53 (all far from significant). Per the abstract's own statistical standard for "reaches", `realism_only` reaches the ceiling at least as much as `full_attrforge` does.

This is the headline empirical claim. It is the first sentence of the "What we find" paragraph in the abstract, and it is paragraph (ii) of three substantive observations in Section 7.1. It does not survive a direct lookup against the data file the paper itself points to.

FIX:
1. Either remove "only" from the abstract sentence (then the claim becomes "AttrForge is one of several iterated conditions that match the ceiling"), or
2. Reframe the comparison around the right contrast (`full_attrforge` vs `full_classic` specifically, not vs all iterated conditions), or
3. Change "every iterated condition" to "the multi-critic conditions we explicitly compare", which would require explicit acknowledgment that `realism_only` and `diversity_only` are not part of the headline comparison even though they appear in the same table.

**BL-NEW-2. The augmentation gap between `full_attrforge` and `full_classic` is not statistically distinguishable from zero at ANY of the six real-train sizes.**

WHERE: Abstract "What we find" paragraph: "the seven-critic loop is the only iterated condition that reaches the real-only macro F1 ceiling ($0.893$)... the three-critic baseline plateaus at $0.872$ and stays there" implies an empirically meaningful difference of 0.021 between `full_attrforge` and `full_classic` from n = 15 onward. Section 7.1 paragraph 2 repeats this framing as substantive observation (ii).

WHY: I computed paired-t and Wilcoxon for `full_attrforge` minus `full_classic` at each n directly from `scarce_real.json`:

| n | mean diff | sd_diff | paired-t p | Wilcoxon p | 95% boot CI | direction AF>FC |
|---|-----------|---------|-----------|-----------|-------------|-----------------|
| 5  | -0.087 | 0.135 | 0.224 | 0.375 | [-0.193, +0.016] | 2/5 |
| 10 | -0.023 | 0.152 | 0.756 | 0.875 | [-0.139, +0.096] | 2/5 |
| 15 | +0.001 | 0.162 | 0.986 | 1.000 | [-0.125, +0.128] | 2/5 |
| 20 | +0.021 | 0.134 | 0.740 | 0.750 | [-0.085, +0.128] | 2/5 |
| 25 | +0.021 | 0.134 | 0.740 | 0.750 | [-0.085, +0.128] | 2/5 |
| 30 | +0.021 | 0.134 | 0.740 | 0.750 | [-0.085, +0.128] | 2/5 |

NONE of these comparisons are significant at any conventional threshold. Under Bonferroni at alpha = 0.05 / 6 = 0.0083, also none. The direction is +AF at every n in [15, 30] but only 2 of 5 seeds (40%) show AF > FC. The paired difference 0.021 is dwarfed by the within-condition sd (0.075 and 0.134). The 95% bootstrap CI of the difference comfortably includes zero at every n.

The cell-by-cell means in Table 3 happen to land at 0.893 (FA) vs 0.872 (FC) when averaged across the discrete F1 values the 10-item test set can produce. But the per-seed pattern is: seeds 17 and 53 show AF > FC by +0.16; seeds 23 and 41 show AF < FC by -0.107; seed 89 shows tie. The 5-seed mean is dominated by seeds 17 / 53 where AF beats FC, and seeds 23 / 41 where it loses. This is exactly the seed-dependent noise pattern the round-2 review flagged for the isolated-protocol claim, just transposed to the augmentation protocol.

The "AttrForge matches the ceiling; full_classic plateaus" claim is therefore an artifact of:
- 10-item test set produces only 22 distinct F1 values, several of which are common attractors (0.733, 0.893, 1.0)
- 5 seeds is enough that the mean lands on one of these attractors but not enough that the difference is significant

FIX:
1. Report paired-t / Wilcoxon p-values in Table 3 alongside the per-cell means. The abstract should say "the difference between `full_attrforge` and `full_classic` at n >= 15 is not statistically distinguishable from zero at N = 5 (all paired-t p > 0.7); the per-seed direction is split 2/5".
2. Drop the "matches the ceiling vs plateaus" framing. Replace with the actual observable: at the noise floor of a 10-item test set, both conditions land near the ceiling at sufficient n; their separation is within seed variance.

**BL-NEW-3. The Table 3 "plateau at 0.872" is a discretization artifact, not a stable mean.**

WHERE: Section 7.1 paragraph 2 (ii): "the three-critic baseline plateaus at $0.872$ and stays there." Table 3 column `full_classic` shows IDENTICAL means at n = 15, 20, 25, 30 (all 0.872 +/- 0.134).

WHY: The per-seed `full_classic` values from `scarce_real.json` at n = 15, 20, 25, 30 are IDENTICAL ACROSS ALL FOUR SIZES: [0.733, 1.000, 0.893, 0.733, 1.000] (seeds 17, 23, 41, 53, 89 respectively). Likewise, `full_attrforge` per-seed values at n = 20, 25, 30 are identical: [0.893, 0.893, 0.787, 0.893, 1.000]. This means that beyond a certain n, the classifier's prediction on the 10-item test set is INSENSITIVE to additional real training data when 48 synthetic samples are also in the mix. The "plateau" is therefore not a property of the synthetic-data quality of `full_classic`; it is a property of the augmented training set being dominated by 48 synthetic examples + a real subset that the classifier saturates on.

This means the narrative "the three-critic baseline plateaus" suggesting that `full_classic`'s synthetic data has a quality ceiling is misleading. A more honest reading: the augmented classifier doesn't change its predictions as n grows from 15 to 30 because the per-class signal is already saturated.

FIX:
1. Add a sentence to Section 7.1: "The identical per-seed values of `full_classic` at n = 15, 20, 25, 30 indicate that the classifier predictions on the 10-item test set are insensitive to additional real-train examples once n >= 15. The same insensitivity holds for `full_attrforge` from n = 20 onward. The apparent plateau is therefore a property of the saturated downstream classifier rather than a quality difference in the synthetic batches."
2. Drop the "plateaus at 0.872" implication of quality limitation.
3. Run the same experiment with a 50-item or 100-item held-out test (10 items is too small to detect the differences the paper wants to claim).

**BL-NEW-4. The "+0.10 to +0.23 over real-only at n in {5, 10}" claim in the abstract is inaccurate.**

WHERE: Abstract "What we find" paragraph: "At smaller real-train sizes ($n \in \{5, 10\}$) both methods provide large gains over real-only ($+0.10$ to $+0.23$ macro F1)".

WHY: Verified gains per `scarce_real.json`:
- n=5: `full_classic` gain = +0.229, `full_attrforge` gain = +0.142
- n=10: `full_classic` gain = +0.103, `full_attrforge` gain = +0.080

`full_attrforge` at n=10 has a gain of +0.080, which is BELOW the abstract's lower bound of +0.10. The combined range across both methods and both sizes is [+0.080, +0.229], not [+0.10, +0.23]. Either the lower bound is wrong, or the abstract excludes the FA-at-n=10 case from "both methods".

Section 7.1 paragraph 2 (i) says "+0.14 to +0.23 over the real-only baseline" at n=5 (this matches the actual range for n=5 alone: FA=+0.142, FC=+0.229; small floor mismatch of 0.002 is rounding).

So there is a numeric inconsistency between abstract ([+0.10, +0.23]) and Section 7.1 ([+0.14, +0.23]). Both bounds should be derivable from the same data file.

FIX:
1. If the abstract intends to cover n=5 AND n=10, the range should be [+0.08, +0.23] (FA at n=10 is the lower bound).
2. If the abstract intends to cover n=5 only, drop "n in {5, 10}" from the claim.
3. Match Section 7.1 to whichever choice is made above.

### MAJOR

**MA-NEW-1. Confound: few_shot baseline uses a different exemplar pool (8 real) than other iterated conditions (3 real); this is not isolated in the augmentation experiment.**

WHERE: Table 2 footnote: "`few_shot` additionally raises the generator's few-shot pool from 3 to 8 real exemplars." Table 3 includes `few_shot` as a comparator.

WHY: At n = 5, `few_shot` augmentation scores 0.668; `full_classic` scores 0.789 with gain over real-only of +0.229. But `few_shot` uses 1 iteration of 16 synthetic samples generated with EIGHT real exemplars in the few-shot pool, while `full_classic` uses 3 iterations of 16 = 48 synthetic samples generated with THREE real exemplars per iteration. The Section 7.1 narrative reads `full_classic` > `few_shot` as a validation of iteration; but `full_classic` has 48 samples vs `few_shot`'s 16. This is a sample-count confound on top of the few-shot-pool confound.

The cleanest fair comparison `full_classic` vs `few_shot` would require either (a) `few_shot` at 3 iter (= 48 samples) with the 8-exemplar pool, or (b) `full_classic` at 1 iter (= 16 samples) with the 3-exemplar pool. Neither is in the paper.

FIX:
1. Add a "few_shot iter=3" condition (8 exemplars, 3 iterations, no critics) and report. This isolates whether the gain comes from iteration or from the larger exemplar pool.
2. Or add a "1-iteration full_classic" / "1-iteration full_attrforge" to isolate the per-iteration effect.
3. Or acknowledge the confound in Section 7.1: "The comparison `few_shot` vs iterated conditions confounds the exemplar-pool size, the iteration count, and the critic feedback; we do not separate these."

**MA-NEW-2. Could `full_classic` look better with different hyperparameters (more synthetic samples, larger pool)?**

WHERE: The hyperparameter "samples per iteration = 16; 3 iterations" is fixed across all iterated conditions. Section 6.3.

WHY: The augmentation headline rests on the claim that `full_attrforge`'s 48 synthetic samples push the augmented classifier to the ceiling more reliably than `full_classic`'s 48 synthetic samples. But this is at a single point in the (synthetic sample count) x (real-train size) plane. The natural follow-up question is whether `full_classic` would close the 0.021 gap if it generated 96 samples instead of 48, or if it used 8 exemplars per iteration like `few_shot`. The paper does not run these.

If `full_classic` matches the ceiling at 96 samples and `full_attrforge` does so at 48, that would still be an interesting positive finding (cost reduction) but a different headline ("AttrForge reaches the ceiling with fewer samples") and weaker than "AttrForge is the only iterated condition that reaches the ceiling".

FIX:
1. Run one additional condition: `full_classic` at 6 iterations of 16 (96 samples), and compare to `full_attrforge` at 3 iterations of 16 (48 samples). If `full_classic_x2` matches the ceiling, the headline becomes "AttrForge reaches the ceiling with half the synthetic data". If it doesn't, the current framing strengthens.
2. Or explicitly acknowledge in the limitations that hyperparameter sweeps were not done.

**MA-NEW-3. Statistical re-verification: paired-t / Wilcoxon are not reported in Table 3.**

WHERE: Table 3, Section 7.1.

WHY: The abstract's TF-IDF claim is supported with explicit "paired-t p = 0.046". The new headline augmentation claim is supported with NO p-values, no CIs, only +/- std-dev. Given that the abstract specifically says `full_attrforge` "reaches the ceiling" and `full_classic` "plateaus at 0.872", a reader would expect to see a paired-t at n = 20, 25, 30. Per my analysis above, paired-t p = 0.74 at all three sizes; the difference is not significant.

FIX:
1. Add a "paired-t p" column to Table 3 comparing each condition vs real-only and against `full_classic`. The result will be that the augmentation headline does not reach conventional significance at any n.
2. Add explicit text: "The paired difference `full_attrforge` minus `full_classic` is not statistically distinguishable from zero at any n (paired-t p > 0.7 at every size). We report the comparison as a directional / pattern observation rather than as a hypothesis-test conclusion."
3. Or, alternately, frame the augmentation result around the directional observation that `full_attrforge` is the only iterated condition whose POINT ESTIMATES match the ceiling at every n in [20, 30], explicitly disclaiming statistical confidence at N = 5.

**MA-NEW-4. The "matches the ceiling" framing is a weak positive claim when `full_attrforge` is below real-only at lower n.**

WHERE: Section 7.1 paragraphs 2 (ii) and (iii); Abstract "What we find" paragraph.

WHY: Even if we grant that `full_attrforge` matches the ceiling at n >= 20, the augmentation should DOMINATE real-only at small n (where real labels are scarce and the value of synthetic should be highest). At n = 5, `full_attrforge` (0.703) is BELOW `full_classic` (0.789), which itself is below the theoretical ceiling 0.893. At n = 10, `full_attrforge` (0.828) is also below `full_classic` (0.851). So the paper's preferred 7-critic loop:
- Loses to its own 3-critic baseline at the regime where synthetic-data augmentation is most useful (n = 5 with FC at +0.229 gain vs FA at +0.142 gain)
- Ties with the 3-critic baseline at n in [15, 30]
- Matches the ceiling at n >= 20 (where there is no headroom anyway because real-only is already at 0.893)

The "match the ceiling at n >= 20" claim is a positive claim about the regime where the value of synthetic data is LEAST informative (the ceiling is already met by real-only). The actually-most-informative regime (n = 5, n = 10) shows `full_attrforge` UNDERPERFORMING `full_classic` by +0.087 macro F1 at n=5. This is the same direction as the isolated TF-IDF gap, just at smaller magnitude.

FIX:
1. Restructure Section 7.1 around the actually observable pattern: in the scarce-real regime (n = 5 or 10), the 3-critic loop wins; in the saturation regime (n >= 20), both methods land near the ceiling within noise. The headline should be honest about that.
2. Or: reframe as "the seven-critic loop closes the augmentation gap at large n", which is a weaker but defensible positive.

**MA-NEW-5. The classifier choice for the augmentation experiment is not isolated from the augmentation framing.**

WHERE: Section 7.1 uses sentence-transformer + LR. Section 7.2 + 7.3 demonstrate that the diversity-utility comparison is CLASSIFIER-DEPENDENT (TF-IDF favors `full_classic` significantly).

WHY: The augmentation experiment is run on sentence-transformer features only. We know from Section 7.2 / 7.3 that under TF-IDF features the picture inverts. So the headline augmentation claim is implicitly conditional on the choice of downstream classifier. The same multi-classifier comparison should be run for the augmentation protocol (run augmentation with TF-IDF word and TF-IDF char, not just sentence-transformer). This was the core insight of Section 7.3.

If the augmentation comparison under TF-IDF shows `full_classic` >> `full_attrforge` (as the isolated protocol does), then the augmentation headline reduces to "under sentence-transformer features the embedding-based classifier absorbs the diversity that hurts TF-IDF; both methods land near the ceiling at saturating n; the difference between them is not significant".

FIX:
1. Run scarce_real_eval.py with `tfidf_word` and `tfidf_char` features in addition to `st_minilm`. Add a multi-panel Figure 2 (one panel per classifier).
2. State explicitly: "the augmentation headline is conditional on the sentence-transformer classifier choice; under TF-IDF features, the diversity-utility cost of the 7-critic loop is not absorbed by the classifier, and the headline reverses or vanishes."

**MA-NEW-6. The new "absorbs surface diversity" claim is asserted at the end of the abstract without measurement.**

WHERE: Abstract last paragraph: "the GAN-style adversaries inject surface diversity: `full_attrforge` has the highest distinct-$n$ and the lowest self-BLEU-4 among all iterated conditions."

WHY: This is now correctly supported by Table 6. The mechanism claim that the sentence-transformer "absorbs" this diversity while TF-IDF does not absorb it is not directly measured. A cleaner test would be to compute the cosine distance between sentence-transformer embeddings of synthetic vs real, and show that `full_attrforge`'s synthetic samples are closer to real distribution than `full_classic`'s under sentence-transformer features (smaller MMD or larger overlap) but not under TF-IDF. The paper does not run this.

FIX:
1. Add a single direct measurement (MMD between train and synth under each feature space, per condition). If `full_attrforge` lies closer to real under sentence-transformer features but further under TF-IDF, the mechanism claim is supported. If both metrics agree, the "embedding absorbs diversity" framing needs rethinking.
2. Or weaken the mechanism claim to a hypothesis ("we conjecture that the embedding classifier maps the surface variation into its existing real-vs-synth-decision boundary; we do not measure this directly").

### MINOR

**MI-NEW-1.** The abstract's "+0.10 to +0.23 macro F1" range is internally inconsistent with Section 7.1's "+0.14 to +0.23" range. See BL-NEW-4. Trivial to align once data is corrected.

**MI-NEW-2.** Table 3 caption: "The real-only baseline at $n=20$ already reaches the same F1 as $n=30$, indicating that the sentence-transformer classifier saturates on this task at around 20 real examples." This is true and useful disclosure. Could be expanded to acknowledge that this saturation explains why the "ceiling match" claim is hard to read as a meaningful AttrForge property.

**MI-NEW-3.** Figure 2 caption: "The seven-critic loop (red) tracks the real-only baseline at $n \in \{15, 20, 25, 30\}$ and reaches the $0.893$ ceiling at $n \geq 20$." The same claim issue as BL-NEW-1: `realism_only` also tracks and exceeds the ceiling but is not drawn on the plot (the plot only shows `full_classic` and `full_attrforge`). Add `realism_only` and `diversity_only` lines, or restrict the caption to the explicit-comparison pair.

**MI-NEW-4.** Section 7.5 paragraph 1 talks about "the per-iteration view (the classifier trained on a single iteration's 16-sample batch rather than the 48-sample pool)". The framing is precise. But Figure 4 caption still says "Both are far from the chance target: the loop makes samples more detectable, not less, but the GAN-style adversaries soften the trend." This is the per-iteration framing of the cumulative-pool result; the realism discriminator metric in Table 4 (0.93 vs 0.90) is the final-iteration value, and it does NOT speak to whether the trend "softens" through iterations. The "softens the trend" wording is loose.

**MI-NEW-5.** Wilcoxon p-values are reported alongside paired-t in Section 7.2 ("Wilcoxon $p = 0.88$" for ST) and Section 7.3 ("paired-t $p = 0.66$, Wilcoxon $p = 0.75$"). The newer numbers (paired-t p = 0.82 in Section 7.2 abstract; paired-t p = 0.66 in Section 7.3 paragraph) DISAGREE. Looking carefully at Section 7.2 vs 7.3: Section 7.2's "$p = 0.82$" is the ST gap under the abstract's main comparison; Section 7.3's "$p = 0.66$" is the same comparison referenced one section later. These should match exactly. They appear to be from different runs (the 0.66 may be the round-3 N = 3 result; the 0.82 the N = 5 result; the author left both).

FIX: Recompute once on the N = 5 data and use a single number. Currently the abstract and Section 7.3 paragraph have different p-values for what should be the same test.

**MI-NEW-6.** Section 7.5: "the seven-critic loop's effect is back-loaded. Across 5 seeds, `full_attrforge`'s per-iteration F1 rises from iter-0 ($0.25 \pm 0.09$) through iter-1 ($0.37 \pm 0.16$) to iter-2 ($0.49 \pm 0.19$), the highest mean of any iterated condition". Verified from per_iter.csv. But the +/- 0.19 at iter 2 means the 95% CI on the iter-2 mean is +/- 0.17 with N = 5, i.e., [0.32, 0.66]. The "highest mean of any iterated condition" framing should acknowledge that with this CI width, the rank-1 cannot be statistically separated from the rank-2 condition.

**MI-NEW-7.** "The seven-critic loop's instruction set accumulates over iterations, so its early iterations resemble the baseline while later ones diverge." This is a hypothesis about mechanism. The data shows non-monotone per-iteration F1, consistent with several mechanisms. The "accumulating instruction set causes back-loading" interpretation could be tested by a 6-iteration run (do iters 3-5 continue rising or fall back?). Not in scope, but worth a sentence about the mechanism being conjectural.

**MI-NEW-8.** Section 8 (audit): "We omit Coverage AUROC (saturates at $1.00$ for every condition) ... both are reported in the released artifacts but do not differentiate conditions on this task." This is honest. But the issue is whether the AUROC = 1.0 is a real coverage failure or an overfit logistic regression on a small training set. Round-3 M-NEW-6 asked for a shuffle-label or k-fold check; this was not done. The sentence "do not differentiate conditions" is correct empirically; the deeper question "is this a coverage failure or a classifier artifact" remains unaddressed.

**MI-NEW-9.** Section 7.4 caption: "The non-iterated baselines (`naive` and `few_shot`) score higher on distinct-$n$ because they have not been narrowed by iterative prompt updates". The observation is correct in the table. But Table 6 footnote could acknowledge that `naive` and `few_shot` have 16 samples vs iterated 48; smaller batches by nature have higher distinct-n at fixed lexicon size. So the comparison isn't pure.

## Specific concern: does the augmentation headline survive scrutiny?

### Are the n in [20, 30] ceiling-match results statistically meaningful?

No. Paired-t and Wilcoxon at every n in [20, 25, 30] for `full_attrforge` vs `full_classic` give p > 0.74 (see BL-NEW-2 table above). The 95% bootstrap CI of the difference comfortably includes zero at every n. Per-seed direction is split 2/5. Under Bonferroni at alpha = 0.05/6 = 0.0083 (six tests), no comparison comes close. At N = 5 with a 10-item test set producing 22 distinct F1 attractor values, the headline difference 0.872 vs 0.893 (which is 1/30 of a misclassification on a 10-item, 5-class, support-2-per-class test) is on the order of one held-out sample changing class.

### Is "matches the ceiling" a sufficient positive claim when `full_attrforge` ≠ real-only at lower n?

No. The scarce-real regime (n = 5, n = 10) is where synthetic data should be most valuable. At these sizes, `full_attrforge` (+0.142 gain at n=5, +0.080 at n=10) loses to `full_classic` (+0.229 at n=5, +0.103 at n=10). The 7-critic loop "matches" the ceiling specifically at the regime where the real-only baseline ALREADY matches the ceiling (real-only at n=20 is 0.893, identical to the n=30 saturation). So the headline claim is a positive statement about the regime where synthetic data does not measurably help; in the regime where it does help, the 7-critic loop is the worse choice. This is a weak positive claim. See MA-NEW-4.

### Could `full_classic` look better with different hyperparameters?

Yes, probably. The 0.021 gap between `full_classic` (0.872) and `full_attrforge` (0.893) at n in [20, 30] is one held-out sample on the test set. With more synthetic samples (e.g., 6 iterations instead of 3 = 96 samples) or a larger few-shot pool (8 instead of 3), `full_classic` would likely close this gap. The paper does not run these. See MA-NEW-2.

### Are there confounds?

Yes, two important ones.
1. `few_shot` uses 8 real exemplars vs all iterated conditions use 3. This contaminates any cross-row comparison in Table 3 that includes `few_shot`.
2. `naive` and `few_shot` use 1 iteration of 16 samples vs iterated conditions 3 iterations of 16 = 48 samples. This is 3x the synthetic sample count. The Section 7.1 narrative compares iterated vs non-iterated without explicitly accounting for this.

The intra-iterated comparison (`full_classic` vs `full_attrforge` vs `self_critique` vs `realism_only` vs `diversity_only`) is internally consistent on samples and exemplars (3 exemplars, 48 samples each). But ALL of those comparisons are statistically non-significant per the analysis in MA-NEW-3.

The classifier confound (MA-NEW-5) is most serious: the headline is conditional on sentence-transformer features. Section 7.3 of the same paper documents that switching to TF-IDF flips the conclusion.

## Statistical re-verification

### Paired-t and Wilcoxon for `full_attrforge` vs `full_classic` augmentation at each real-train size

Per-seed values from `scarce_real.json`:
- Seed order: 17, 23, 41, 53, 89
- `full_classic` at n=15..30: identical [0.733, 1.000, 0.893, 0.733, 1.000] for every n
- `full_attrforge` at n=20..30: identical [0.893, 0.893, 0.787, 0.893, 1.000] for every n

| n | mean diff (AF - FC) | sd(diff) | paired-t | p_t | Wilcoxon p | 95% boot CI | dir AF>FC |
|---|--------------------|---------|----------|-----|------------|-------------|-----------|
| 5  | -0.087 | 0.135 | -1.44 | 0.224 | 0.375 | [-0.193, +0.016] | 2/5 |
| 10 | -0.023 | 0.152 | -0.33 | 0.756 | 0.875 | [-0.139, +0.096] | 2/5 |
| 15 | +0.001 | 0.162 | +0.02 | 0.986 | 1.000 | [-0.125, +0.128] | 2/5 |
| 20 | +0.021 | 0.134 | +0.36 | 0.740 | 0.750 | [-0.085, +0.128] | 2/5 |
| 25 | +0.021 | 0.134 | +0.36 | 0.740 | 0.750 | [-0.085, +0.128] | 2/5 |
| 30 | +0.021 | 0.134 | +0.36 | 0.740 | 0.750 | [-0.085, +0.128] | 2/5 |

None of these tests are significant. Under Bonferroni at alpha = 0.05/6 = 0.0083, no comparison comes close.

### Paired-t for `full_attrforge` augmentation vs real-only at n = 30

| n | mean diff (AF - RO) | sd(diff) | paired-t | p_t | Wilcoxon p | 95% boot CI | dir AF>RO |
|---|--------------------|---------|----------|-----|------------|-------------|-----------|
| 5  | +0.142 | 0.182 | +1.74 | 0.156 | 0.188 | [-0.022, +0.259] | 4/5 |
| 10 | +0.080 | 0.061 | +2.94 | 0.043 | 0.063 | [+0.035, +0.125] | 5/5 |
| 15 | +0.055 | 0.106 | +1.16 | 0.313 | 0.313 | [-0.031, +0.128] | 4/5 |
| 20 | +0.000 | 0.107 | +0.00 | 1.000 | 1.000 | [-0.085, +0.085] | 2/5 |
| 25 | +0.035 | 0.145 | +0.54 | 0.621 | 1.000 | [-0.064, +0.168] | 1/5 |
| 30 | +0.000 | 0.075 | +0.00 | 1.000 | 1.000 | [-0.064, +0.064] | 1/5 |

`full_attrforge` augmentation at n = 30 is INDISTINGUISHABLE from real-only at n = 30: p = 1.0, mean diff = 0.000, CI [-0.064, +0.064]. By direction, AF beats RO in 1/5 seeds, ties in 3/5, loses in 1/5 (seed 41 always shows AF below RO). The "matches the ceiling" claim at n = 30 is empirically correct (means agree to 4 decimals) but it is also correct that adding 48 synthetic samples is NEUTRAL relative to using only the 30 real samples. This is the same as saying "synthetic data adds no measurable value when n_real reaches the saturation point".

The same analysis for `full_classic` vs real-only at n = 30: mean diff = -0.021, paired-t p = 0.74. Not different from zero.

### Compare AttrForge vs real-only at n = 10 (the most positive case)

`full_attrforge` augmentation at n = 10 vs real-only at n = 10: paired-t p = 0.043, Wilcoxon p = 0.063, 5/5 directional, CI [+0.035, +0.125]. This is the strongest positive statistical finding for AttrForge augmentation in the entire scarce_real experiment. The abstract does mention this as part of "+0.10 to +0.23" gain at small n; the actual measured gain at n=10 is +0.080 with N=5.

### Compare `full_classic` vs real-only at n = 5

`full_classic` augmentation at n = 5: mean diff = +0.229, paired-t p = 0.057, Wilcoxon p = 0.063, 5/5 directional. Approaching significance and consistently above real-only across all seeds. This is the strongest single-condition positive finding for augmentation in the experiment. The paper does report this as part of the +0.14 to +0.23 range at n=5, but it pairs it with `full_attrforge`'s weaker +0.142. By per-condition statistical strength, the augmentation experiment's WINNER is `full_classic` at n=5, not `full_attrforge` at any n.

### Confirm or refute the "AttrForge matches the ceiling" claim with proper hypothesis tests

REFUTED in the form stated in the abstract ("the only iterated condition that reaches"). Three lines of evidence:
1. `realism_only` exceeds 0.893 at all of n=20, 25, 30 (means 0.904, 0.904, 0.925); not just `full_attrforge`.
2. The difference between `full_attrforge` and `full_classic` at n in [15, 30] is not significant under paired-t (p > 0.7) or Wilcoxon (p >= 0.75) at any size.
3. The "match" at n=30 means `full_attrforge` is indistinguishable from real-only (p = 1.0), i.e., adding 48 synthetic samples has no measurable effect on the held-out F1 at this n.

The defensible weaker version: at n >= 20 on this task with N = 5 seeds and a 10-item test, `full_attrforge` is among the iterated conditions whose mean F1 is statistically indistinguishable from the real-only baseline. So is `realism_only`, `diversity_only`, and `full_classic`. So is `few_shot`. The point estimates differ by less than a single held-out sample.

## Writing and clarity issues

- The four-paragraph abstract format (problem / what we do / what we find / what we additionally contribute) is a real improvement over previous rounds; readers can find the headline quickly. Good change.
- Section 7.1 paragraph 2 lists three patterns (i), (ii), (iii). Pattern (ii) "AttrForge reaches the real-only ceiling; full_classic does not" is the load-bearing one and is the one that does not survive scrutiny. Either reframe or run more seeds.
- The contrast between "matches the ceiling" (the new positive claim) and "0.14 macro-F1 cost under TF-IDF" (the previous round's negative finding, demoted to Section 7.2) is somewhat awkward. The paper is making two claims: (A) under augmentation with ST features, AttrForge matches the ceiling, (B) under isolated TF-IDF train-on-synth, AttrForge loses 0.14 to the baseline. Both can be true simultaneously, but the paper's framing oscillates between "this is the headline now" and "the TF-IDF gap was the headline before". A reader could finish the paper without knowing what the authors think the main contribution is.
- Section 7.5 is titled "Realism trajectory" but discusses both the realism discriminator and the per-iteration downstream F1. Two different metrics; recommend splitting into 7.5 (realism discriminator across conditions) and 7.6 (per-iteration downstream F1 dynamics).
- Section 9.6 has a subtle inconsistency: it positions against NanoFlux and Verbalized Sampling but does not engage with how this paper's main finding (an augmentation-only-statistically-distinguishable gain over baseline at n=10) compares to those works' reported effect sizes.
- The "Code is open-source under MIT at github.com/ApartsinProjects/PromptForge" link is good. Worth verifying that the repository actually mirrors the submission's manifest (a TMLR reviewer following the link should reach a clean reproducer).

## Overall recommendation

**MAJOR REVISIONS.**

The paper has improved substantively since round 2. The N=3 -> N=5 seed expansion strengthens the TF-IDF significance from p=0.09 to p=0.046; the direct diversity measurements (distinct-n, self-BLEU-4) add an external check that BL4-of-round-2 specifically requested; the prose unification across abstract / Sections 9, 10, 11 has been completed; the BibTeX and HTML titles match. These are non-trivial improvements that demonstrate good-faith engagement.

However, the v1.4 pivot to the augmentation headline introduces a NEW set of statistical and conceptual problems that are at least as serious as the round-2 issues they replaced. The two central claims in the new abstract (AttrForge is "the only iterated condition" reaching the ceiling; `full_classic` "plateaus at 0.872 and stays there") do not survive direct verification against `scarce_real.json`:
- `realism_only` exceeds the ceiling at all three sizes (n = 20, 25, 30) by larger margins than `full_attrforge` does
- The per-seed `full_classic` values are identical at n = 15, 20, 25, 30, demonstrating that the "plateau" is a property of the saturated classifier on 10-item test, not a property of the synthetic-data quality
- The 0.021 difference between `full_classic` and `full_attrforge` at n in [15, 30] is not statistically distinguishable from zero under paired-t (p > 0.7) or Wilcoxon (p >= 0.75) at any size
- The directional split is 2/5 seeds in favor of AF at n in [15, 30], not 3/5 or higher
- The strongest positive statistical finding in the entire augmentation experiment is actually `full_classic` vs real-only at n=5 (p=0.057, 5/5 directional, +0.23 gain), which the paper underplays

The pattern is: the paper found a partial positive (AttrForge's point estimate matches the ceiling where `full_classic`'s does not), but it pushed that observation up to a "the only iterated condition" headline that the data does not support, because three OTHER iterated conditions (realism_only, diversity_only, few_shot) also reach or exceed that point.

What I would request from a fourth revision:
1. Restate the augmentation headline as: "at n >= 20, all iterated conditions are within 0.05 macro F1 of the real-only ceiling on this task with N=5 seeds and a 10-item test set; the differences are not statistically distinguishable." Frame `full_attrforge` matching the ceiling as a positive but not unique finding. Remove the "only" claim.
2. Repeat the scarce-real experiment with a 50-item or 100-item held-out test set. The 10-item test set is too small to resolve the 0.02 gap the abstract leans on; the discretization to ~22 distinct F1 attractor values is responsible for the "plateau" appearance.
3. Run the scarce_real experiment with TF-IDF features (matching Section 7.2 / 7.3 protocol) and report all three classifiers. If the augmentation headline does not survive the TF-IDF check, demote it; if it does, the headline becomes "AttrForge augmentation reaches the ceiling on stronger downstream classifiers and not on TF-IDF" which is a more honest cross-classifier story.
4. Reframe the Section 7.1 (i) "synthetic data is most valuable when real is scarce" observation around its actual strongest result: `full_classic` augmentation at n=5 delivers +0.229 F1 over real-only with 5/5 directional support. The 7-critic loop's gain (+0.142 at n=5) is weaker. This is the same direction as the isolated-protocol TF-IDF gap, just smaller.
5. Either run a larger-budget condition (more iterations or more samples per iteration) for `full_classic` to test whether the 0.021 ceiling gap closes with more synthetic data, or acknowledge in the limitations that the augmentation comparison is at a single sample budget.

I would not recommend reject. The paper has substance: the multi-critic framework is a sensible engineering contribution; the post-hoc audit machinery is genuinely useful; the multi-classifier insight (Section 7.3) is paper-worthy on its own; and the direct diversity measurements (Section 7.4) cleanly show that the GAN adversaries do their stated job. The augmentation experiment as such is also good experimental design; it is the headline phrasing that overreaches the data.

The minimum I would expect from a revision: the abstract and Section 7.1 either drop the "only" claim and the "plateaus at" framing in favor of the directional / point-estimate observation, OR the experiment is extended to support the stronger claim (multi-classifier augmentation, larger test set, more seeds, or hyperparameter sweep).

## Top 5 fixes ranked by impact

1. **(BL-NEW-1, BL-NEW-2, MA-NEW-3)** Restate the augmentation headline. The current abstract claim "the seven-critic loop is the only iterated condition that reaches the real-only macro F1 ceiling (0.893) at every real-train size from 20 to 30; the three-critic baseline plateaus at 0.872 and stays there" is false (realism_only exceeds 0.893 at all three sizes) and not statistically distinguishable (paired-t p > 0.74 at every n). Replace with: "at n >= 20, point estimates of `full_attrforge` augmentation match the real-only ceiling; the 0.021 difference vs `full_classic` is not statistically distinguishable at N = 5 seeds (paired-t p > 0.7 at every n)".

2. **(MA-NEW-5)** Run the augmentation experiment with all three downstream classifiers (TF-IDF word, TF-IDF char, sentence-transformer), matching the Section 7.3 protocol. The current single-classifier augmentation result is implicitly conditional on the embedding feature space, which Section 7.3 itself documents as the regime where the diversity-utility cost vanishes.

3. **(BL-NEW-3, MA-NEW-2)** Repeat the scarce-real experiment with a 50+ item held-out test set, or add a higher-sample-count condition (e.g., `full_classic` at 96 samples) to test whether the "plateau" is a real per-condition ceiling or a saturation artifact of the 10-item test. Currently the 22 distinct F1 attractor values produced by 10-items-x-5-classes-x-balanced-support drive the "plateau at 0.872" appearance.

4. **(MA-NEW-4)** Reframe Section 7.1 paragraph (i). The strongest single-condition statistical result in the scarce_real experiment is `full_classic` augmentation at n=5 (gain +0.229, 5/5 directional, paired-t p = 0.057). The 7-critic loop's gain at the same point is +0.142. Frame the scarce-real regime around its actual winner, not around the "AttrForge matches the ceiling at saturating n" claim. The "match" at saturating n is a positive observation but it is statistically indistinguishable from "real-only also matches the ceiling at saturating n", which is uninformative.

5. **(MI-NEW-5, MI-NEW-2, BL-NEW-4)** Numerical consistency sweep. The abstract's "+0.10 to +0.23 macro F1" claim does not match Section 7.1's "+0.14 to +0.23" or the underlying data (FA at n=10 is +0.08). The ST p = 0.82 vs p = 0.66 inconsistency between abstract and Section 7.3 must be resolved (re-derive on the N=5 data and use one number). Table 3 should include paired-t / Wilcoxon p-values, not just per-cell means and standard deviations.
