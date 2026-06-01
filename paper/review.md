# TMLR Adversarial Self-Review: AttrForge

## 1. Summary

The paper proposes AttrForge, a multi-objective prompt optimization framework for LLM-based synthetic data generation. The core technical contribution is the transfer of four mode-collapse defenses from the GAN literature (Pack Discriminator, Mode-Seeking ratio, Mode Hunter with persistent banned-phrasing memory, Coverage Hole Finder via density-ratio estimation) into the prompt-debugging setting. The framework augments three baseline LLM critics with these four batch-level adversaries, all routing structured complaints into a prompt updater. Empirical claims are validated on a single customer-support intent task using a deterministic simulator backend (the live-LLM run failed due to OpenAI quota limits) across seven ablation conditions with a single seed.

## 2. Strengths

- Clean architectural separation: the seven critics conform to the same "named-complaint" interface, which makes the ablation table conceptually clean (one bool flag per critic).
- The mapping of four specific GAN defenses (PacGAN, MSGAN, ban-list training, density-ratio coverage) to prompt-space is a genuinely interesting reframing, and the persistent banned-phrasing library ("immune memory") is a natural and reasonable addition.
- Mode-Seeking ratio with a per-attribute sensitivity matrix is a low-cost, model-free signal that directly tests attribute responsiveness; the per-attribute breakdown is a nice diagnostic.
- The code is well structured, readable, and uses Pydantic models with on-disk JSONL persistence. The run-directory layout is clearly documented.
- The simulator backend is honestly labeled as a "harness validator", not as a substitute for real LLMs.
- The Coverage Hole Finder grounds a fuzzy "missing mode" complaint in concrete real exemplars; the design is principled (density-ratio estimation with a logistic regression classifier).
- All seven baseline conditions share the exact same seed, dataset, and metrics; only `enable_*` flags differ. A legitimately clean ablation contract.
- The deterministic verification of Mode Hunter findings via substring counting before adding to the banned library is a sensible guard against LLM hallucination.

## 3. Weaknesses

### Blockers (must fix before any acceptance)

**B1. The headline empirical claim is contradicted by the actual data.** Table 2 reports all iterated conditions (including `full_attrforge`) at downstream macro-F1 = 0.787. This is the cumulative metric (training on all 48 samples). However, per-iteration F1 for `full_attrforge` is: iter 0 = 0.300, iter 1 = 0.169 (worse than naive 0.300), iter 2 = 0.693. `full_classic` reaches iter 2 = 0.787. So at iteration 2, `full_attrforge` is worse than `full_classic` by 0.094 F1. This is hidden by cumulative pooling. **Fix**: report both cumulative and per-iteration F1 honestly; acknowledge the per-iter degradation; or remove the equivalence claim.

**B2. The realism objective got worse over iterations, not better.** Section 3 defines the realism objective with chance level 0.5 as the target. The actual data shows discriminator accuracy went 0.708 → 0.833 → 0.833 for `full_attrforge` and similarly for `full_classic`. Realism degraded by 0.125. The paper does not mention this. **Fix**: acknowledge that the loop is moving the discriminator AWAY from chance in this simulation; explain why (the simulator's deterministic discriminator detects formal openers and tic phrases; the updater's added clauses do not suppress them effectively).

**B3. The "30 requests before quota error" claim is not supported by logs.** `experiments/_run.log` shows the quota error fired on the first request of every condition (5 separate 429 errors, each within iteration 1 of a fresh condition). **Fix**: produce evidence or rephrase to "the API was unavailable before any iteration completed".

**B4. Single seed, 10 test examples, no error bars.** F1 differences of 0.094 on 10 examples are noise. The Pack accuracy 0.75 has `n_comparisons = 4` (per `config.sim.yaml`), making the metric quantized to 5 values. **Fix**: run 3-5 seeds, compute mean ± std, report Pack accuracy null distribution (e.g., from `pack(real, real)`).

### Major issues

**M1. Section 3 says seed set size is `N_r ∈ [50, 200]`, but the actual seed set is 40 examples** (30 train, 10 test). Section 6.1 admits 40. **Fix**: align Section 3 with reality.

**M2. The mode-seeking ratio is presented as a diagnostic but its scale is uninterpretable.** Section 7.2 admits "0.22 is informative only relative to the real-data ratio" but the real-data ratio is never reported. **Fix**: compute and report on real seed set; use ratio of synth-ms to real-ms.

**M3. The simulator updater is buggy: it appends literally duplicate instruction blocks across iterations.** `full_classic` v2 prompt is the v1 prompt with the entire Style and constraints block appended verbatim a second time. **Fix**: either fix the simulator updater to deduplicate or shorten the claim about "structurally richer prompts".

**M4. The "samples_per_iteration = 16" used in the run is undocumented in the paper.** The configured default is 12; the actual run used 16. **Fix**: state batch size in Section 6.3 and include exact CLI in an appendix.

**M5. The "harness validation" framing does not justify the methodological claim.** With 10 test examples, anything above F1 = 0.8 is at the noise floor. The argument "the loop reveals what other axes cannot" requires a real-LLM run where the downstream metric is not saturated. **Fix**: either run real-LLM experiment, or reduce the claim to "in our simulator, the GAN-style adversaries detect a measurable axis invisible to per-sample critics by construction".

**M6. The "differential firing" claim in Section 7.2 is tautological.** Pack and Mode-Seeking columns say "off" for every condition where the flag is false. So "only `full_attrforge` produces Pack accuracy" is true by construction. **Fix**: post-hoc, run Pack/MS/Hunter/CoverageHole as AUDITORS on every condition's final batch and report differential numbers.

**M7. The Coverage Hole Finder reports AUROC = 1.0 for both `diversity_only` and `full_attrforge`.** AUROC = 1.0 means perfect separation real vs synthetic (a strong negative signal). The paper does not discuss it. A healthy loop should drive AUROC toward 0.5. **Fix**: report and interpret AUROC; show iteration-over-iteration AUROC.

**M8. No related-work coverage of recent prompt-optimization work.** Missing: APE (Zhou et al. 2023), OPRO (Yang et al. 2024), PromptBreeder (Fernando et al. 2023), EvoPrompt (Guo et al. 2024), DSPy (Khattab et al. 2024), TextGrad (Yuksekgonul et al. 2024). **Fix**: add Section 2.5 with these citations and a one-paragraph differentiation.

### Minor issues

**M9.** The 280-word budget in Section 4.4 is inconsistent with the code (UPDATER_SYSTEM says ~250). Pick one.

**M10.** "Three backends" in Section 6.2 misses the `echo` backend. Either drop or add it.

**M11.** Figure 1 reuses the hero image. A real architecture box-and-arrow diagram would be more informative.

**M12.** Pack discriminator with `n_comparisons = 4` quantizes accuracy to 5 values. Reported 0.75 is one trial above chance.

**M13.** "Symmetry with the baseline critics" claim breaks for Mode-Seeking (returns scalar + vector, no named complaint).

**M14.** Section 7.5's "non-monotone trajectory" rationalizes bad data. The iter-1 drop happens for ALL iterated conditions; it is a simulator artifact, not a loop signature.

**M15.** Section 4.3 mentions sentence-transformer embeddings but the actual run uses TF-IDF (`use_embeddings: false`).

## 4. Empirical concerns

- **E1**. Saturation framing is artifactual on a 10-item test set.
- **E2**. Pack accuracy 0.75 has no null comparison (pack(real, real)).
- **E3**. Per-attribute sensitivity matrix is computed but never shown.
- **E4**. Mode Hunter library growth = 0 → 2 → 3 → 3 should be reported with `banned_phrasings_new` per iteration.
- **E5**. "Combination coverage" reported as 0.8 does not match Section 3's β-weighted formula.
- **E6**. Cumulative downstream comparison is unfair (naive uses 16 samples, others 48). Subsample 16 from cumulative for fair comparison.

## 5. Methodological gaps (claim vs code)

- **G1**. Section 3 defines a β-weighted diversity scalar that the code never computes.
- **G2**. Coverage-gap formula uses only attribute pairs (no higher-order), undocumented.
- **G3**. Updater diversity block is a single JSON blob, not a named-section list.
- **G5**. Coverage hole exemplars never reach the simulator generator (no sim updater route for `coverage_hole_block`).
- **G6**. `attribute_sensitivity` is populated but never consumed by the simulator updater.
- **G7**. Verifier's "difficulty hard" rule (`difficulty == "hard" and keyword_hits >= 1 and len < 80` ⇒ fail) is simulator-specific and not disclosed in paper.

## 6. Writing and clarity

- **W1**. Abstract claim "invisible to the baseline" is trivially true (baseline doesn't run the metric).
- **W2**. Asymmetry: 3 motivating failures → 7 critics. Make the 2x2 grid explicit.
- **W4**. Tables 1 and 2 use "off" for both "disabled" and "not measured". Distinguish.
- **W5**. "Immune memory" over-branding repeats 3 times. Pick one place.
- **W6**. "Pareto-style compromise" overstates joint constraint satisfaction.
- **W8**. Reference list (9 items) is light for a bridging paper.

## 7. Missing related work

- Prompt optimization: APE, OPRO, EvoPrompt, PromptBreeder, DSPy, TextGrad.
- LLM-as-judge for synthetic data: West et al. 2022, Bonifacio et al. 2022 (InPars), Meng et al. 2022 (SuperGen), Ye et al. 2022 (ProGen).
- Mode-collapse measures: FID (Heusel et al. 2017), Precision/Recall for distributions (Sajjadi et al. 2018), Density/Coverage (Naeem et al. 2020).
- Synthetic-data quality detection: G-Eval (Liu et al. 2023), Self-consistency (Wang et al. 2023).

## 8. Overall recommendation

**Major revisions** (TMLR rubric: "claims do not match evidence"). The headline experimental claims (per-iter equivalence in Table 2, realism convergence, "30 requests") cannot be verified from the artifacts. Path to acceptance: real-LLM run with multiple seeds, properly sized test set, post-hoc audit of GAN metrics across all conditions, re-grounded claims that match artifacts.

## 9. Top 5 fixes ranked by impact

1. **Run live-LLM experiment with 3+ seeds and 50+ test examples, report per-iter and cumulative with std deviations.** Without this, no quantitative claim is credible.
2. **Run Pack / Mode-Seeking / Mode-Hunter / Coverage-Hole as AUDITORS on every condition's final batch.** Converts Table 2's tautological "off" into real differential measurements.
3. **Acknowledge and fix simulator pathologies**: discriminator going up, updater duplication, per-iter F1 regression. Either fix the sim or restate Section 7 honestly.
4. **Add Section 2.5 on prompt optimization** (APE, OPRO, EvoPrompt, DSPy, TextGrad). Reposition contribution as multi-objective structured-complaint feedback.
5. **Fix factual inconsistencies**: seed-set size (40 not 50-200), "30 requests", 280 vs 250 word budget, samples-per-iter undocumented, "off" ambiguity in Table 2.
