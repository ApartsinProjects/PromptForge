# TMLR Third-Round Review: "Adversarial Prompt Debugging for LLM Synthetic Data Generation" (v0.7, formerly v0.6 with title "It Depends on the Classifier: ...")

Reviewer: senior reviewer
Round: 3 (post multi-classifier addition and statistical rewrite)
Verdict: MAJOR REVISIONS (see Section "Overall recommendation")

## Summary of changes since round 1

The authors restructured the framing around a multi-classifier downstream evaluation (TF-IDF word, TF-IDF char 3-5, sentence-transformer MiniLM) and rewrote the abstract with honest paired statistics (paired-t p=0.09, Wilcoxon p=0.25, 95% bootstrap CI [-0.27, -0.07] on the TF-IDF diff; +0.038 with CI [-0.07, +0.18] on the sentence-transformer diff). Several round-1 blockers are now fixed: figures are sourced from main_run_002 live-LLM data, the future-dated ref-gradcollide citation is removed, the abstract no longer contains the false insufficient_quota caveat, and Section 5.3 honestly discloses min_repeats = 1. However, the rewrite happened in the abstract, Section 1, and Section 7.2 only; Sections 7.1, 7.3, 9, 10, and 11 still carry residual claims, numbers, and framing from the prior simulator-headline era, producing a paper that is now internally inconsistent in a different way than before.

## What round-1 issues are now resolved

- BL1 (figure-vs-claim contradiction): figures are now sourced from `main_run_002` live-LLM data; Figure 5 is no longer `sim_run_001_per_class_f1.png`.
- BL3 (Mode Hunter min_repeats misrepresentation): Section 5.3 now honestly states "we use m=1 in our headline run because n_batch=16 is small enough that requiring >=2 co-occurrences misses most tics in expectation". Honest disclosure.
- BL5 (downstream classifier confound): a multi-classifier evaluation (Section 7.2, Table 2b, Figure 3) has been added.
- M1 (insufficient_quota caveat in Section 6.2): the caveat in Section 6.2 has been replaced with an accurate "Which backend produced which numbers" block describing the live-LLM run honestly. The caveat survives in Section 10 limitations item 1 as orphan text (see new MAJOR M1 below).
- M8 (future-dated ref-gradcollide citation): removed. Section 9.6 now reads "Concurrent preprint work characterizes a related failure mode... we discuss the empirical anchor for our named-complaint design once the preprint is stable." Adequate.
- M12 (mode hunter count reported as 12 instead of 11.67): now reported as "11.67 +/- 0.58" in Table 2 and abstract. Honest.
- M10 (audit Pack accuracy SD = 0 across 3 seeds): audit SDs are now non-zero (e.g., 0.20, 0.06, 0.13, 0.16, 0.10) in Table 3, indicating the audit-RNG re-seeding now actually varies across the 3 outer seeds. Fixed.

## What round-1 issues remain

- BL2 (paired-difference statistics on TF-IDF gap). The abstract and Section 1 now correctly report paired-t p=0.09 and the bootstrap CI. But Section 7.1 paragraph 3 still claims the gap is "more than three times the cross-seed standard deviation". The cross-seed STD is 0.05 (full_classic) and 0.06 (full_attrforge); the gap 0.187 IS approximately 3x those numbers. But this is a WITHIN-condition SD, not the paired-difference SD that the test should compare against. The paired-difference SD is 0.106, so the gap is 1.76x that, NOT 3x. Section 7.1 should report the paired statistics, not the "3x cross-seed std" framing which conflates two different uncertainty quantifications. The abstract is honest; Section 7.1 is not.
- BL4 (direct diversity measurement). Still not done. The paper claims Mode Hunter accumulates 11.7 LLM-tic phrasings as evidence the GAN adversaries are doing their job, but does not report distinct-n, self-BLEU, or any external direct-diversity measure on the live-LLM batches. The "11.7 banned phrasings" is also dependent on min_repeats=1; with the looser threshold, single-occurrence outputs are counted. A direct surface-diversity measurement that does NOT depend on the loop's own critics would substantially strengthen the argument.
- M2 (internal data reference Section 7.3): "at iteration 2 full_attrforge sits slightly below full_classic in the per-iteration metric (0.69 vs 0.79 on the same seed mean)" is still in Section 7.3. I verified per_iter.csv: iter-2 macro F1 is full_classic = 0.299 +/- 0.079, full_attrforge = 0.444 +/- 0.107 - that is, full_attrforge is ABOVE full_classic at iteration 2, not below. The 0.69 vs 0.79 numbers do not appear anywhere in the artifacts. This is a stale number from a previous run that should have been updated when the figures were resourced from main_run_002.
- M3 (iter-1 dip explanation as "simulator's generator"). Figure 4 caption still says "All conditions show a transient iter-1 dip, which we attribute to the simulator's generator responding to early prompt-updater instructions in ways that initially hurt label discriminability before consolidating." But Figure 4 is now sourced from main_run_002 (live gpt-4o-mini), not the simulator. The caption rationalization is incoherent with the data source.
- M5 (Table 1 "the only change is which critics are enabled" claim, conflating naive/few-shot/iterated). The Table 1 caption is now more honest: "the conditions differ in which critics are enabled and in two non-critic settings: naive and few_shot use 1 iteration of 16 samples (16 total); the other five use 3 iterations of 16 samples (48 total). few_shot additionally bumps the generator's few-shot pool from 3 to 8 real exemplars." Adequate, partial fix.
- M6 (realism numbers contradiction simulator vs live): now resolved by reporting only live-LLM values in Section 7, but the orphan paragraph at line 471 ("on this N_test = 10 task the downstream metric saturates the moment a condition gets 48 training samples regardless of which critics are enabled") still claims saturation, which is contradicted by the headline 0.58 vs 0.40 gap in the SAME section. This is internally inconsistent.

## New issues introduced in v0.6 / v0.7

### BLOCKERS

**BL-NEW-1. The paper has a split personality between sections.**

WHERE: Abstract and Section 1 vs Sections 7.1, 7.3, 9, 10, 11.

WHY: The new abstract and Section 1 are honest and statistically sound. They report:
- TF-IDF gap with paired statistics (p=0.09, bootstrap CI [-0.27, -0.07], directional 3/3)
- Sentence-transformer gap as indistinguishable from zero (p=0.66, CI [-0.07, +0.18])
- Honest framing: "the cost is real under a low-capacity reader and not detectable under a stronger one"
- The honest finding "is not that the tradeoff reverses, but that it vanishes"

But the body of the paper still asserts the opposite framing:
- Section 7.1 paragraph 3: "a 0.18 gap that is more than three times the cross-seed standard deviation" (statistically wrong; see BL2 above)
- Section 7.2 H3 heading: "The tradeoff is classifier-dependent (and reverses with sentence-transformer features)" (asserts reversal that the same section's body explicitly disavows)
- Section 7.2 paragraph body: explicitly admits "the paired difference is statistically indistinguishable from zero"
- Figure 3 caption: "The ranking of full_classic vs full_attrforge reverses between TF-IDF... and sentence-transformer embeddings" (asserts reversal that is not statistically supported)
- Contributions item 4: "with sentence-transformer embeddings, the 7-critic loop produces the best F1 of all conditions" (true by point estimate; statistically tied with at least 4 other conditions)
- Section 9.1: "The multi-seed simulator results establish four claims..." (numbers 0.69 +/- 0.06 for full_attrforge realism do not match either main_run_002 or sim_run_002; appear to be stale)
- Section 9.2: "We also do not claim that the simulator's behavior generalizes to live LLM behavior" (the headline IS the live-LLM result)
- Section 10 Limitations item 1: "Live-LLM results pending. Our headline numbers come from a deterministic prompt-sensitive simulator..." (directly contradicts abstract and Section 1)
- Section 11 Conclusion: "Through multi-seed simulator-validated experiments..." (same)
- Section A BibTeX: still has the old title "More Critics, Worse Data?..."

A reader following the abstract through to the Discussion gets two different papers. The Discussion/Limitations/Conclusion describe a simulator paper that does not exist anymore.

FIX:
1. Sweep Sections 9, 10, 11 to replace "simulator" framings with live-LLM framings.
2. Delete Section 10 item 1 ("Live-LLM results pending").
3. Rewrite Section 9.1 to summarize the live-LLM findings and their statistical confidence.
4. Update Section 11 to match the abstract.
5. Update BibTeX title and HTML/OG title to match the actual paper title shown in the H1.

**BL-NEW-2. The "reversal" framing in the contributions, Section 7.2 heading, Figure 3 caption, and Section 1 is statistically unsupported and directly contradicted by the body text.**

WHERE: Contributions item 4; Section 7.2 H3 heading; Figure 3 caption.

WHY: The abstract correctly says "the honest finding is not that the tradeoff reverses, but that it vanishes". The body of Section 7.2 correctly says "the paired difference is statistically indistinguishable from zero". Yet:
- The H3 heading is "The tradeoff is classifier-dependent (and reverses with sentence-transformer features)".
- Figure 3 caption asserts "The ranking of full_classic vs full_attrforge reverses between TF-IDF (where the 3-critic loop wins) and sentence-transformer embeddings (where the 7-critic loop wins)."
- Contributions item 4 says: "with sentence-transformer embeddings, the 7-critic loop produces the best F1 of all conditions. We argue this is the right baseline for train-on-synth / test-on-real reporting: multi-classifier evaluation, not single-classifier evaluation."

This last item, in particular, mixes a defensible claim (multi-classifier evaluation is the right protocol) with an indefensible one (the 7-critic loop produces the best F1 of all conditions). Per the per-seed data I recomputed (paper/_per_seed_f1.json), the sentence-transformer condition rankings are:

| condition | st_minilm F1 mean +/- sd | per-seed |
|-----------|--------------------------|----------|
| full_attrforge | 0.711 +/- 0.083 | [0.713, 0.793, 0.627] |
| realism_only | 0.680 +/- 0.199 | [0.648, 0.893, 0.500] |
| diversity_only | 0.676 +/- 0.200 | [0.500, 0.633, 0.893] |
| full_classic | 0.673 +/- 0.129 | [0.533, 0.787, 0.700] |
| few_shot | 0.631 +/- 0.103 | [0.733, 0.527, 0.633] |
| self_critique | 0.578 +/- 0.135 | [0.500, 0.733, 0.500] |
| naive | 0.454 +/- 0.098 | [0.567, 0.408, 0.387] |

full_attrforge vs realism_only: paired-t p=0.69
full_attrforge vs diversity_only: paired-t p=0.84
full_attrforge vs full_classic: paired-t p=0.66
full_attrforge vs few_shot: paired-t p=0.48

None of these comparisons are significant. The mean rank-1 of full_attrforge under sentence-transformer is plausibly random noise from N=3 seeds with high per-seed variance. The "produces the best F1 of all conditions" claim should be deleted.

FIX:
1. Change Section 7.2 H3 heading to "The TF-IDF tradeoff vanishes under stronger downstream classifiers".
2. Change Figure 3 caption to: "The TF-IDF gap is consistent across all three seeds; the gap on sentence-transformer features is not statistically distinguishable from zero. The character n-gram classifier shows an intermediate gap." Drop the word "reverses".
3. Change contributions item 4 to: "We document that the apparent diversity-utility tradeoff on TF-IDF features vanishes on embedding features, and argue that multi-classifier evaluation is the right protocol for train-on-synth / test-on-real reporting."

**BL-NEW-3. Section 9.1 contains specific numbers (0.69 +/- 0.06 realism for full_attrforge; 0.72 +/- 0.10 for full_classic) that do not correspond to any artifact in the repository.**

WHERE: Section 9.1 paragraph 1.

WHY: I checked main_run_002_aggregated/summary.json: discriminator_accuracy for full_classic = 0.917 +/- 0.144 and full_attrforge = 0.833 +/- 0.167. I checked sim_run_002 aggregated table (not in the new repo as far as I saw): the v1 review reported sim full_classic realism = 0.72, full_attrforge = 0.69. The numbers 0.69 / 0.72 quoted in Section 9.1 appear to be the SIMULATOR numbers from the previous draft, not updated to match main_run_002. They also do not match Table 2 in the current paper (which reports 0.92 and 0.83 for the same metric, same conditions).

This is the same class of bug as round-1 M2: stale prose numbers that didn't get updated when the data source changed.

FIX: Replace the 0.69 / 0.72 with the actual main_run_002 numbers (0.83 / 0.92), or delete the sentence as redundant with Table 2.

**BL-NEW-4. The Section 7.3 per-iteration claim "(0.69 vs 0.79 on the same seed mean)" contradicts the actual iter-2 data in per_iter.csv.**

WHERE: Section 7.3 paragraph 1.

WHY: From experiments/main_run_002_aggregated/per_iter.csv, iter-2 macro F1 across 3 seeds is full_classic = 0.299 +/- 0.079 and full_attrforge = 0.444 +/- 0.107. The paper claims 0.69 vs 0.79, with full_attrforge below full_classic. Both the direction and the magnitudes are wrong:
- Direction: full_attrforge is ABOVE full_classic at iter 2 (the per-iter dynamics tell a story OPPOSITE to the cumulative-pool dynamics).
- Magnitude: 0.30 / 0.44, not 0.69 / 0.79.

This is a serious internal inconsistency since the paper draws a substantive conclusion ("we do not claim full_attrforge dominates full_classic on downstream F1 in this setting") from a number that is wrong.

The accurate per-iter story is that full_attrforge has a low iter-0 and iter-1 (0.241, 0.247) and a strong iter-2 recovery (0.444), while full_classic stays flat across iterations (0.249, 0.253, 0.299). Cumulatively (training on all 48 samples) the rankings reverse. This is actually a more interesting story; the paper just needs to tell it correctly.

FIX: Replace the iter-2 numbers with the actual per_iter.csv values, and rewrite the paragraph accordingly.

### MAJOR

**M-NEW-1. Section 10 limitations contains orphan simulator-era text contradicting the abstract.**

WHERE: Section 10 item 1: "Live-LLM results pending. Our headline numbers come from a deterministic prompt-sensitive simulator... our live-LLM attempt was halted by an insufficient_quota error..."

WHY: This is the exact paragraph round-1 review M1 flagged as misrepresenting what was run. It was removed from Section 6.2 (good) but survives verbatim in Section 10 (not good). The abstract says the headline numbers come from gpt-4o-mini main_run_002.

FIX: Delete item 1 entirely. Replace with: "Single-domain validation, N=3 seeds, N_test=10. The honest paired statistics produce wide bootstrap CIs (e.g., +/- 0.10 width); a larger budget would tighten the conclusions but would not change the qualitative finding."

**M-NEW-2. Section 7.2 has two H3 sections both numbered 7.2.**

WHERE: H3 line 429 "The tradeoff is classifier-dependent (and reverses with sentence-transformer features)" and H3 line 475 "Realism trajectory" both have `<span class="subsection-num">7.2</span>`.

WHY: Numbering bug. Compounds the reader confusion already created by BL-NEW-1.

FIX: Renumber the second one to 7.3 (and shift the existing 7.3 to 7.4, 7.4 to 7.5).

**M-NEW-3. The Pack null reference disagrees between Figure 7 caption and Table 3 caption.**

WHERE: Figure 7 caption: "PackAcc_null = 0.50 in this domain". Table 3 caption: "PackAcc_null = 0.54 in this domain".

WHY: I checked main_run_002_aggregated/summary.json: null_pack_accuracy_real_vs_real = 0.542. So the Table 3 caption is right (within rounding) and Figure 7 caption is wrong. The per-seed audit summaries show seed-17 null = 0.50, seed-23 null = 0.625, seed-41 null = 0.50, so the value depends on the run. Either way, one of the two captions is inconsistent.

FIX: Update Figure 7 caption to 0.54 to match Table 3. Or report both per-seed and aggregated nulls.

**M-NEW-4. The orphan paragraph at line 471 contradicts Section 7.1.**

WHERE: Section 7.2 last paragraph (line 471): "Two observations from Table 2 and Figure 2. First, on this N_test = 10 task the downstream metric saturates the moment a condition gets 48 training samples regardless of which critics are enabled, so the downstream metric alone cannot distinguish AttrForge from other iterated baselines."

WHY: Section 7.1 reports a 0.58 vs 0.40 macro F1 gap between full_classic and full_attrforge; both are iterated conditions with 48 samples. So the metric DOES distinguish them. This sentence is residual prose from the simulator era when the iterated conditions DID saturate at the same F1. Now they do not.

Also: "the only loop-internal metric where full_attrforge differs from full_classic by more than one seed-std is the realism discriminator accuracy: 0.69 +/- 0.06 vs 0.72 +/- 0.10. The full AttrForge stack moves the realism objective 0.03 closer to chance than the three-critic baseline, while exactly matching it on attribute fidelity (1.00) and downstream metrics." Numbers are stale (real numbers are 0.83 +/- 0.17 vs 0.92 +/- 0.14; attribute fidelity is 0.06 / 0.10 NOT 1.00 / 1.00; downstream metrics do NOT match per Section 7.1).

FIX: Delete the paragraph entirely; it is residue from the simulator draft.

**M-NEW-5. The classifier-dependent framing leaks into the abstract metadata.**

WHERE: HTML title (line 6): "Adversarial Prompt Debugging for LLM Synthetic Data Generation". OG/Twitter titles same. H1 (line 34) same. But meta-description and twitter:title in the previous v0.6 said "It Depends on the Classifier".

WHY: It looks like v0.7 stripped the classifier-dependent framing from the title but kept it in the abstract and Section 7.2 (with the reversal claim). The title sounds generic; the abstract is doing the heavy framing work. Combined with BL-NEW-1, this makes it hard for a reader to tell what the paper is arguing.

FIX: Decide on a title that matches the actual finding. Suggested: "When Does Adversarial Prompt Debugging Help? A Classifier-Dependent Diversity-Utility Tradeoff". Or, more honestly given the statistics: "On the Classifier Dependence of Train-on-Synth Evaluation: A GAN-Style Multi-Critic Case Study".

**M-NEW-6. The Coverage Hole Finder AUROC = 1.00 on every condition is reported as a negative finding, but the paper does not interrogate whether it is a coverage failure or a small-N classifier artifact.**

WHERE: Section 8 takeaway 3.

WHY: With 30 real + 16 to 48 synthetic, an unregularized logistic regression on TF-IDF features will almost surely separate real from synthetic perfectly even when the two distributions overlap. The paper says "whether this reflects fundamental separability of gpt-4o-mini output from human customer-support text, or an artifact of small-N logistic regression with fully-flexible vocabulary, remains open." But the artifact-vs-real question is testable cheaply: shuffle labels and check if AUROC drops to 0.5. Cross-validation on 5 folds would also tell you. The paper acknowledges the question; it does not answer it.

FIX: Run the shuffle-label or k-fold check; report. If AUROC drops with shuffled labels, the separation is real and the paper's framing is supported. If it doesn't drop much, the AUROC = 1.0 is a small-N artifact and the "Coverage Hole Finder remains at 1.00 on every condition" is uninformative.

**M-NEW-7. The contributions list still claims "the 7-critic loop produces the best F1 of all conditions" (under sentence-transformer features).**

WHERE: Contributions item 4.

WHY: See BL-NEW-2 above. By point estimate full_attrforge ST F1 = 0.711 IS the highest mean, but the pairwise differences are all non-significant. The "best of all conditions" claim is technically defensible by point estimate but the paper has just spent the abstract telling the reader the differences are within noise. Internal inconsistency between the abstract and the contributions list.

FIX: Reword item 4 to match the abstract: "we report a TF-IDF gap that vanishes under embedding features, with a wide CI that does not support a reversal claim; multi-classifier evaluation is the right protocol."

**M-NEW-8. Section 4.4 still asserts "Table 0 is the core conceptual claim of our architecture: structured named complaints admit locally targeted rewrites that a scalar reward does not."**

WHERE: Section 4.4.

WHY: Round-1 M7 flagged this: there is no controlled experiment comparing scalarized critic outputs against named-complaint outputs. The paper still presents the claim as "core" without empirical support.

FIX: Either run a controlled scalarized-baseline ablation, or restate the claim as "we found a structure that worked; we did not test against scalarized alternatives in this paper."

**M-NEW-9. Section 9.6 still has "we discuss the empirical anchor for our named-complaint design once the preprint is stable."**

WHERE: Section 9.6.

WHY: This is now a placeholder for the future-dated ref-gradcollide citation. The sentence is awkward as published prose. Either cite a real, verified anchor for the "multi-objective text-gradient methods degrade" claim, or drop the sentence entirely.

FIX: Drop the sentence or replace with a real citation.

**M-NEW-10. Section 9.1 still describes "simulator results" with simulator-era numbers.**

WHERE: Section 9.1, full paragraph.

WHY: BL-NEW-1 already covers the high-level inconsistency; this is the specific number bug. The four-claims structure of Section 9.1 was written for the simulator paper. It needs to be rewritten for the live-LLM paper.

FIX: Replace with a four-claims summary of the live-LLM findings: (i) the harness is correct; (ii) the TF-IDF gap is real and directional across 3/3 seeds; (iii) the gap vanishes on stronger embedders; (iv) two honest negative findings about the loop-internal metrics.

**M-NEW-11. Bug audit (sanity_audit.md) flags "Bug C. Mode Hunter signal sparsity at batch=16" as "PARTIALLY FIXED (min_repeats=1 + real LLM produces more tics)".**

WHERE: paper/sanity_audit.md.

WHY: This is honest internal documentation but the paper does not surface the partial-fix status. With min_repeats=1, a "tic" is anything the Mode Hunter LLM happens to flag in any single sample. The 11.67 banned phrasings count is therefore an LLM-judge measurement, not a structural mode-collapse measurement. The Mode Hunter library size is being reported as evidence the GAN adversaries are doing their job, but the loose threshold makes the metric tautological with whatever the LLM judge feels like flagging.

FIX: Either run with min_repeats=2 and report whatever number comes out (likely small), or in the paper acknowledge: "the 11.67 banned phrasings figure is a count of LLM-judge-flagged distinct items at min_repeats=1; with the canonical min_repeats=2 threshold from the GAN-defense literature, this count would drop substantially. We use the relaxed threshold because n_batch=16 is too small to expect 2+ exact repetitions of structural tics."

### MINOR

**N-NEW-1.** The audit_summary.json per-seed null_pack_accuracy varies (0.50, 0.625, 0.50). The aggregated value 0.542 is a mean of these three. The audit captioning should reflect that the null itself has seed-to-seed variance, not present 0.54 or 0.50 as a single number.

**N-NEW-2.** Figure 7 caption says the null is 0.50. Table 3 caption says 0.54. Pick one.

**N-NEW-3.** The paper still uses the phrase "an `harness validator`" (Section 6.2 caveat block heading) - actually it's now "a harness validator" which is right. Round-1 N8 fixed.

**N-NEW-4.** The HTML title and Twitter card title were both updated to "Adversarial Prompt Debugging for LLM Synthetic Data Generation" but the H1 visible to readers is the same. The og:image was updated to `figures/main_run_002_multi_classifier.png`. Internal consistency.

**N-NEW-5.** The "Reproduce with" CLI in Section 6.3 now correctly invokes `--backend openai` and `--run-id main_run_002`. Round-1 M13 fixed.

**N-NEW-6.** The DOCX downloadable still labels itself as v0.7 but the body of the paper may still have older numbers (I did not verify the .docx; the HTML is the authoritative version).

**N-NEW-7.** Section 9.2 says "all iterated conditions saturate at the same F1, which we attribute to the noise floor of a 10-item test set". This is the simulator framing; with live LLM, iterated conditions do NOT saturate at the same F1 (0.58 vs 0.40 vs 0.41 vs 0.33 vs 0.35). The sentence needs revision.

**N-NEW-8.** The sentence-transformer score table (Table 2b) shows realism_only and diversity_only at 0.68 (basically tied with full_attrforge 0.71). The paper could honestly say: "Under embedding features, all iterated conditions cluster between 0.58 and 0.71, with N=3 seeds insufficient to rank them."

**N-NEW-9.** The bibliographic year 2026 + Section A BibTeX title still reading "More Critics, Worse Data?..." while the H1 reads "Adversarial Prompt Debugging..." is a versioning inconsistency that any TMLR copy editor will flag.

**N-NEW-10.** Section 7.4 (Per-class F1) describes "(single representative seed)" but does not say which seed. Per-class F1 should be averaged over seeds, or at minimum labeled by seed.

**N-NEW-11.** The sentence-transformer column in Table 2b has standard deviations as wide as 0.20 (realism_only and diversity_only). The paper does not comment on the fact that these wide error bars effectively undermine any condition-ranking claim under ST features.

**N-NEW-12.** The per-class F1 figure (Figure 5) is described as "single representative seed" but the figure filename `main_run_002_per_class_f1.png` does not encode which seed. Use a seed-specific filename or aggregate.

**N-NEW-13.** Round-1 N2: "Anonymous" author lists still appear for ref-paretoprompt, ref-strongcollapse, ref-nanoflux, ref-syntheggs, ref-mgtfilter, ref-improveddre. The bib_validation.md companion noted these need verification. None of the v0.7 entries verify the author lists.

## Statistical robustness re-check

I recomputed per-seed F1 from the live-LLM artifacts (paper/_per_seed_f1.json) and ran paired tests on the headline comparisons. Results:

**TF-IDF word features (full_attrforge vs full_classic):**
- per-seed paired diffs (AF - classic): [-0.227, -0.267, -0.067]
- mean diff = -0.187, sd = 0.106
- paired t = -3.06 on df=2; p = 0.093 two-sided
- Wilcoxon signed-rank p = 0.25 (the actual minimum p for N=3 is 0.25)
- 95% bootstrap CI [-0.267, -0.067]

The paper's reported statistics (p=0.09, CI [-0.27, -0.07]) match. Honest reporting. The signed direction is consistent across all 3 seeds (3/3 directional).

**TF-IDF char 3-5 features (full_attrforge vs full_classic):**
- diffs: [-0.047, -0.130, -0.007]
- mean diff = -0.061, sd = 0.063
- paired t = -1.69, p = 0.234
- 95% bootstrap CI [-0.130, -0.007]

Mid-significant; the gap is smaller than on word features but still directional 3/3.

**Sentence-transformer features (full_attrforge vs full_classic):**
- per-seed paired diffs (AF - classic): [+0.180, +0.007, -0.073]
- mean diff = +0.038, sd = 0.130
- paired t = +0.51, p = 0.66 two-sided
- Wilcoxon p = 0.75
- 95% bootstrap CI [-0.073, +0.180]

This is not a "reversal". This is statistical noise dominated by seed 17 (+0.18). Seeds 23 and 41 show diffs near zero or negative. The CI comfortably includes zero. The paper's abstract is honest about this; the title-of-section, figure caption, and contributions item 4 are not.

**full_attrforge vs other strong baselines under sentence-transformer:**
- vs realism_only: p = 0.69 (diffs: +0.066, -0.10, +0.127)
- vs diversity_only: p = 0.84 (diffs: +0.213, +0.16, -0.267)
- vs few_shot: p = 0.48 (diffs: -0.02, +0.267, -0.007)

None significant. full_attrforge is statistically tied with realism_only, diversity_only, full_classic, and few_shot under sentence-transformer features. The "best F1 of all conditions" claim in contributions item 4 is true by point estimate, false by paired statistics. With N=3 seeds and the observed variance, the rank-1 position of full_attrforge cannot be statistically distinguished from rank 2, 3, 4, or even 5.

**Multiple-testing context:**

The paper tests 1 hypothesis (full_attrforge vs full_classic) across 3 classifiers. Even unadjusted alpha = 0.05 is not crossed for any classifier:
- TF-IDF word p=0.09
- TF-IDF char p=0.23
- sentence-transformer p=0.66

Under Bonferroni at alpha/3 = 0.017, none are significant. The honest statistical statement is: at N=3, none of the 3 classifier comparisons reaches conventional significance, but the TF-IDF word gap is directional across all 3 seeds (binomial p = 1/8 = 0.125 for the sign test). The "directional 3/3" framing in the abstract is the strongest defensible claim and the paper should consistently use it.

**Verdict on the headline empirical claim:**

The TF-IDF gap is real-but-underpowered. The sentence-transformer "gap" is well within noise. The honest framing in the abstract ("the cost is real under a low-capacity reader and not detectable under a stronger one... the honest finding is not that the tradeoff reverses, but that it vanishes") is correct and statistically defensible. The reversal-asserting framings in Section 7.2 heading, Figure 3 caption, and contributions item 4 are not.

## Overall recommendation

**MAJOR REVISIONS.**

This is the second round of major revisions. The authors have made real progress on the round-1 blockers: the multi-classifier evaluation directly addresses BL5; the statistical framing in the abstract directly addresses BL2; the figure-source contradiction in BL1 is fixed; the Mode Hunter min_repeats issue in BL3 is now honestly disclosed; the future-dated citation in M8 is removed; and the audit-RNG seeding issue in M10 is fixed. These are non-trivial corrections that show the authors took round 1 seriously.

However, the v0.6 / v0.7 rewrite was applied only to the abstract, Section 1, and Section 7.2 body. Sections 7.1, 7.3, 9, 10, 11, and the BibTeX/title metadata were not synchronized. The result is a paper that reads as two stitched-together documents:
- Document A (abstract, Section 1, Section 7.2 body): honest, statistically careful, claims the TF-IDF gap is real-but-underpowered and the embedding gap vanishes.
- Document B (Sections 7.1, 7.3, 9, 10, 11, BibTeX): the simulator-era paper, with stale numbers, the "3x cross-seed std" framing that BL2 flagged, the "reversal" framing that the abstract disavows, the "Live-LLM results pending" limitation that contradicts the headline, and the "simulator-validated experiments" Conclusion.

A TMLR reviewer reading the paper end-to-end will be confused by where the actual evidence base lives. The simplest reading is that the authors did the work but didn't propagate the rewrite. Any single fix above is bounded; collectively they require a full prose-sweep, which is what major revisions is for.

The disposition I would request: another revision round with a unified narrative. I think this paper can become acceptable. The empirical contribution (a classifier-dependent diversity-utility cost, demonstrated rigorously even at the underpowered N=3 budget) is genuinely interesting and the post-hoc audit protocol is a real methodological improvement. The paper just needs to be internally consistent before TMLR can accept it.

I would not recommend reject. The work is substantively in better shape than round 1 suggested. The authors are engaging in good faith with the review feedback. With one careful sweep, the paper becomes acceptable.

## Top 5 remaining fixes ranked by impact

1. **Synchronize the prose across the paper to the abstract's framing (BL-NEW-1).** Drop "simulator" from Sections 9.1, 9.2, 10 item 1, 11. Drop "Live-LLM results pending" from Section 10. Update BibTeX and HTML/OG title to match H1.

2. **Remove the "reversal" framing wherever it appears (BL-NEW-2).** Update Section 7.2 H3 heading, Figure 3 caption, and contributions item 4 to say "the gap vanishes" not "the gap reverses". The paper's own statistical analysis supports the first, not the second.

3. **Update Section 7.1 paragraph 3 to use paired statistics, not "3x cross-seed std" (BL2 residual).** Quote the paired-t p=0.09 and the CI [-0.27, -0.07] from the abstract. Stop comparing the difference to a within-condition SD.

4. **Fix Section 7.3 per-iteration numbers (BL-NEW-4).** Replace "0.69 vs 0.79 on the same seed mean" with the actual iter-2 values from per_iter.csv: full_classic 0.30 +/- 0.08, full_attrforge 0.44 +/- 0.11. Rewrite the paragraph: at iter 2, full_attrforge OVERTAKES full_classic per-iteration (interesting per-iter dynamics), but the cumulative-pool comparison reverses this.

5. **Run a controlled diversity measurement on the live-LLM batches (BL4 residual).** Distinct-n, self-BLEU, embedding-cluster count. Without an external direct-diversity measure, the "GAN adversaries inject surface variation" half of the mechanism is gated on the Mode Hunter library size which is itself measured by an LLM judge at a relaxed threshold (min_repeats=1).

## Confidential note to action editor

The authors are engaging seriously. v0.7's abstract is the most honest abstract I've seen on a paper at this evidence level; explicitly admitting the embedding-classifier "reversal" is within noise (p=0.66, CI [-0.07, +0.18]) shows discipline that most ML authors would walk away from. The multi-classifier experiment is a substantive new piece of work, not a cosmetic patch. The post-hoc audit machinery is genuinely useful and would be cited by other groups trying to do honest synthetic-data evaluation.

But the manuscript still has the seams showing. The body of the paper (Sections 7.1, 7.3, 9, 10, 11) and the bibliographic metadata still read like the previous (simulator-era) paper. A TMLR reviewer can read the abstract and walk away thinking "this is a solid honest paper"; a TMLR reviewer who reads the body will get whiplash from contradictions like Section 10 item 1's "Live-LLM results pending" or Section 9.1's stale 0.69 +/- 0.06 numbers. The two voices in the manuscript currently undermine each other.

I think the right disposition is one more major-revisions round. The required work is bounded: a careful prose-sweep, a few number updates from per_iter.csv, deletion of the orphan limitation item, BibTeX title sync, and one optional additional experiment (Section 7.4 N-NEW-6 about Coverage Hole AUROC, or BL4 about direct diversity). After that, this is an acceptable TMLR paper. Without it, a different reviewer might recommend reject on the internal-inconsistency grounds alone, which I think would be unfair given the underlying work.

I would also be charitable about the N=3 budget. The honest CI width is what it is; the conclusions are correspondingly bounded. If the authors are willing to retitle the paper as a methodological contribution about multi-classifier evaluation protocols, with the empirical case study acknowledged as exploratory, the publication threshold lowers substantially. The "diversity-utility tradeoff" main claim is too strong for the data; the "classifier-dependence of train-on-synth evaluation" claim is well-supported.

My priors going into round 3 were "this will be accepted or rejected based on whether they did the multi-classifier work and the prose-fixes". They did the multi-classifier work; they partially did the prose fixes. Verdict: major revisions, with high expectation of acceptance after one more round if the synchronization fixes land.
