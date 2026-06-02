# TMLR Second-Round Review: "More Critics, Worse Data? GAN-Style Adversaries Expose a Diversity-Discriminability Tradeoff in LLM Synthetic Data Generation"

Reviewer: senior reviewer
Round: 2 (post pivot to the diversity-discriminability framing)
Verdict: MAJOR REVISIONS (see Section "Overall recommendation")

## Summary

The paper proposes SynSmith, a multi-critic prompt-debugging loop for LLM-based synthetic data generation, with four batch-level adversaries (Pack Discriminator, Mode-Seeking, Mode Hunter, Coverage Hole Finder) adapted from the GAN literature. After a pivot from the original "SynSmith improves over baselines" framing, the paper now presents a "diversity-discriminability tradeoff" empirical finding: a 3-critic loop achieves 0.58 plus/minus 0.05 downstream macro F1 on a customer-support intent task with `gpt-4o-mini`, while a 7-critic loop drops to 0.40 plus/minus 0.06 across 3 random seeds. The authors also describe a post-hoc adversary audit protocol that runs every critic against every condition's final batch. The pivot is intellectually honest in spirit, and the engineering of the harness is solid; however, the manuscript in its current form has critical internal inconsistencies, statistical claims that the data do not support, and methodological misrepresentations that block acceptance.

## Strengths

- The engineering of the seven-critic loop is genuinely careful. The named-complaint contract (Section 4.4, Table 0) is conceptually clean, and the on-disk per-iteration manifest format (loop.py, line 7-23) supports reproducible re-analysis.
- Mapping four specific GAN defenses onto prompt space (PacGAN -> Pack Discriminator, MSGAN -> Mode-Seeking ratio, ban-list training -> Mode Hunter, density-ratio -> Coverage Hole Finder) is a nontrivial reframing. The 2x2 grid framing (plausible homogeneity vs attribute-deafness, with vs without cross-iteration memory) in Section 5 is a useful organizing schema.
- The post-hoc adversary audit (Section 8, posthoc_audit.py) genuinely addresses last round's tautological-firing concern: it re-runs every adversary against every condition's pooled final batch with fresh RNG state and a real-vs-real null reference, decoupling "which critic ran in the loop" from "which critic observed the batch". The decision to re-instantiate critics per condition (line 105-108 of posthoc_audit.py) is a real methodological improvement over the v1 audit.
- The deterministic substring verification of Mode Hunter LLM claims (mode_hunter.py:155-156) is a sensible guard against LLM hallucination of phantom tics.
- The Coverage Hole Finder's exemplar-injection design (coverage_hole.py:88-89) is principled: it converts a fuzzy "missing modes" feedback into concrete real exemplars the updater can ground its rewrite in.
- Mode-Seeking with a per-attribute sensitivity matrix (mode_seeking.py:127-141) is a low-cost, model-free diagnostic for attribute-deafness.
- The previous-round revisions are visible: APE/OPRO/EvoPrompt/PromptBreeder/DSPy/TextGrad/GEPA are now cited (Section 2.2); InPars/SuperGen/ProGen/West are cited (Section 2.1); FID/Sajjadi/Naeem are cited (Section 2.4); 3 seeds are now reported instead of 1.
- The recent (2025) related work scout (Section 2.4b: Verbalized Sampling; Section 2.4c: Model Collapse; Section 2.5b: Improved DRE; Section 2.5c: Scendi) is creditable and shows real reading of the area.
- The honest negative findings (mode-seeking ratio constant at ~0.23x, coverage AUROC at 1.00, attribute-match below 10%) are reported rather than buried, in the spirit of the negative-result triage protocol the project description endorses.

## Weaknesses

### BLOCKERS

**BL1. The headline empirical claim is inconsistent with the table that supposedly supports it.**

WHERE: Abstract ("a 3-critic loop achieves 0.58 plus/minus 0.05 downstream macro F1 [...] while adding our 4 GAN-style adversaries drops it to 0.40 plus/minus 0.06") and Section 1 paragraph 2 (same numbers), versus Table 2 in Section 7.1, which reports `full_classic` macro F1 = 0.76 plus/minus 0.05 and `full_attrforge` macro F1 = 0.76 plus/minus 0.05 (identical to 4 decimals).

WHY: I verified by reading the experiment artifacts. The 0.58/0.40 numbers come from `experiments/main_run_002_seed{17,23,41}/` (the live `gpt-4o-mini` run, mean macro F1 across seeds: full_classic 0.585, full_attrforge 0.398). The 0.76/0.76 numbers in Table 2 come from `experiments/sim_run_002_aggregated/table.csv` (the simulator run). The figures Figure 2, Figure 3, Figure 4, Figure 6, Figure 7 are all named `sim_run_002_*`, that is, they are simulator data. The paper appears to have pivoted the abstract to live-LLM numbers but left the results section reporting simulator numbers. The two narratives are silently glued together.

Additionally, Section 6.2 explicitly states: "The headline results in this paper use the `sim` backend" and the inset caveat claims "Our live-LLM run with `gpt-4o-mini` via the OpenAI API was halted by an `insufficient_quota` error on the first generator call of every condition". This contradicts (i) the abstract claim, and (ii) the file `experiments/_run_main.log`, which I inspected: it shows main_run_002 ran to completion across all 7 conditions and 3 seeds, with no quota errors, producing 48 samples per iterated condition with realistic gpt-4o-mini output text (e.g. "why is my account showing weird charges? also, had a sushi for lunch today.").

FIX:
1. Decide: either the abstract describes the live-LLM run (in which case replace Table 2 with the live-LLM aggregation, and re-render Figures 2/3/4/6/7 from `main_run_002` instead of `sim_run_002`), OR the abstract describes the simulator run (in which case replace the headline 0.58/0.40 with 0.76/0.76 and reframe the entire "diversity-discriminability tradeoff" claim).
2. Remove the misleading `insufficient_quota` caveat or restrict it to the v1 run that did fail.
3. Provide an aggregated CSV / summary.json for the live-LLM run analogous to `sim_run_002_aggregated/`, so the headline numbers are reproducible from the artifact tree.

**BL2. With N=3 seeds and N_test=10, the headline gap is not statistically significant under a paired test, contradicting the abstract's "larger than the seed standard deviation by a factor of three" framing.**

WHERE: Abstract: "The gap is 0.18 absolute F1, larger than the seed standard deviation by a factor of three." Section 1 paragraph 2 repeats this.

WHY: I verified the per-seed numbers from `experiments/main_run_002_seed{17,23,41}/{full_classic,full_attrforge}/summary.json`:

| seed | full_classic macro F1 | full_attrforge macro F1 | difference |
|------|----------------------|------------------------|------------|
| 17   | 0.627                | 0.400                  | 0.227      |
| 23   | 0.600                | 0.333                  | 0.267      |
| 41   | 0.527                | 0.460                  | 0.067      |

A paired t-test gives t = 3.06 on df = 2, two-sided p = 0.092. Wilcoxon signed-rank gives p = 0.25. Neither reaches conventional significance. The per-condition standard deviation cited in the abstract (0.05 to 0.06) is a between-condition standard deviation that ignores the paired structure of the comparison and ignores that seed 41 shows a tenth of the gap of the other two seeds.

Compounding: N_test = 10 means each macro-F1 point estimate has a Wilson 95% confidence interval of approximately +/- 0.30 for individual proportions; a 0.18 difference is dwarfed by this. With 2 test items per class, any single classification flip changes per-class F1 by 0.33 to 0.5.

The "larger by a factor of three" framing rhetorically conflates within-condition seed std with the paired-difference std. The actual paired-difference SD is 0.106, so the gap of 0.187 is 1.76x that SD, not 3x.

FIX:
1. Report the paired t-test and Wilcoxon p-values directly.
2. Report bootstrap 95% CIs on the difference (paired bootstrap over seeds).
3. Either run 8+ seeds and rerun the test, or temper the abstract: replace "larger than the seed standard deviation by a factor of three" with the actual paired statistics.
4. Explicitly acknowledge the N_test = 10 ceiling. The per-class breakdown (2 items per class) is below any reasonable noise floor for claims of structural mechanism.

**BL3. The "Mode Hunter accumulates 12 distinct LLM-tic phrasings across 3 iterations" claim is materially misrepresented.**

WHERE: Abstract: "the Mode Hunter accumulates 12 LLM-tic phrasings across 3 iterations". Section 5.3: "Each iteration the Mode Hunter is asked to find up to four concrete substrings or structural tics that (i) appear in >= 2 synthetic samples, (ii) appear in 0 real samples".

WHY: I inspected `experiments/main_run_002_seed17/full_attrforge/.../iter_002/mode_hunter.json`. All 12 banned-library entries have `n_synthetic_occurrences: 1`, that is, they appeared in exactly one synthetic sample, not "at least 2" as Section 5.3 promises. Examples (verbatim): "I hope you can help me" (occ=1), "I think it's been hacked..." (occ=1), "I genuinely just want my money back an" (occ=1). Several "tics" are essentially the full text of a single sample.

The discrepancy traces to `examples/customer_support/config.yaml`, which sets `mode_hunter: min_repeats: 1`. The default in `ModeHunterConfig` (mode_hunter.py:71) is `min_repeats: 2`, matching the paper. The config overrides this to 1 with the comment "min_repeats=1 makes Mode Hunter sensitive at batch_size=16". The deterministic verifier (mode_hunter.py:156) then accepts patterns with 1 occurrence.

Effectively, the "12 LLM-tic phrasings" headline statistic is computed under relaxed rules that the paper's method description does not disclose. With the paper's stated rule (>=2 occurrences), the library would be near empty.

FIX:
1. Either change `min_repeats` back to 2 in the config and rerun, OR change Section 5.3 to honestly state "we use `min_repeats = 1` because the batch size is 16, so requiring >=2 occurrences is too restrictive".
2. If sticking with `min_repeats = 1`, acknowledge that a "tic" identified from a single sample is not really a tic in the GAN-defense sense, and reduce the "12 phrasings" claim accordingly.

**BL4. The "increased diversity" half of the diversity-discriminability tradeoff is not supported by direct measurement.**

WHERE: Abstract: "surface diversity measurably increases". Section 1 paragraph 3: "Mode Hunter accumulates 12 distinct LLM-tic phrasings across 3 iterations; banned-phrasing instructions are honored; surface diversity, by any direct measurement, increases."

WHY: I checked the actual diversity-axis numbers in the simulator aggregation (which is the only run with a full audit):
- Mode-seeking ratio (relative to real): full_classic 0.227, full_attrforge 0.227 (Table 3 in paper, also `sim_run_002_aggregated/table.csv`). Identical to 3 decimal places.
- Coverage AUROC: full_classic 1.00, full_attrforge 1.00. Identical.
- Pack accuracy (audit): full_classic 0.44, full_attrforge 0.50. Slight increase, but full_attrforge is *closer to chance* not *further from chance*; the paper itself notes this is the target (Section 8 takeaway 1). It is fine to call this "more diverse" but the claim is then carrying a lot of weight on one metric.
- Near-duplicate rate at theta=0.92: full_classic 0.04, full_attrforge 0.08 (Table 2). full_attrforge is actually 2x more duplicated.

For the live-LLM run there is no aggregated diversity measurement table. I could not find an analog of `table.csv` for `main_run_002`. The closest visible direct-diversity claim is "Mode Hunter accumulates 12 banned phrasings", which is BL3 above.

So at the headline framing level, the "surface diversity, by any direct measurement, increases" sentence is false; in the simulator, two of three audit-grade diversity metrics are flat across full_classic and full_attrforge; one shifts from 0.44 to 0.50 (in the direction the paper says is "more diverse", granted); and near-duplicate rate goes the wrong way. In the live-LLM run, there is no aggregated direct diversity measurement at all.

FIX:
1. Replace "by any direct measurement" with the actual measurements. Acknowledge which metrics move which way.
2. Add a "diversity" measurement that is independent of all of: TF-IDF L2 cosine (which the simulator MS uses), the same Pack discriminator that drives the loop (which is partly tautological even after re-instantiation), and the LLM-as-Mode-Hunter (which is downstream of the LLM bias the paper warns about elsewhere). Suggested independent baselines: Distinct-n, Self-BLEU, embedding-based clustering count, or the Scendi Score (which the paper already cites, ref-scendi).
3. Compute these metrics on the live-LLM batches.

**BL5. The downstream classifier confound is not addressed.**

WHERE: Section 6.4 ("downstream RQ4 metrics (TF-IDF + logistic regression trained on synthetic, evaluated on the held-out real test split)"), Section 7 (no acknowledgement), Section 10 (limitations does not mention this).

WHY: The full causal chain of the "diversity-discriminability tradeoff" mechanism is, on the authors' own framing: critics inject surface variation -> the TF-IDF + LR classifier loses the keyword consistency it depends on -> downstream macro F1 drops. But this depends entirely on the *low-capacity TF-IDF+LR* classifier. The paper's own sanity_audit.md document (which I read; this is a companion document not in the manuscript) explicitly anticipates this: "A more capable downstream classifier (e.g., embedding-based) might invert this finding; we report what we measured."

This caveat does not appear in the manuscript. Yet the entire central finding of the paper rides on it. If a stronger downstream classifier (sentence-transformer + LR, fastText, even tf-idf with character ngrams) inverts the gap, the "tradeoff" disappears. As written, the paper is consistent with two very different worlds: (a) a genuine diversity-discriminability tradeoff that any reasonable classifier would see, and (b) a brittleness of one specific weak classifier to surface variation. The paper presents only the second world's data while making first-world claims.

FIX:
1. Run the downstream evaluation with at least two more classifiers, e.g., sentence-transformer embeddings + LR; fastText; the actual gpt-4o-mini as a zero-shot intent classifier.
2. If the gap survives the more capable classifier, the tradeoff claim strengthens dramatically.
3. If the gap reverses or vanishes, the paper's central claim must be reframed as "TF-IDF + LR is brittle to surface variation", which is a much narrower contribution (and probably not novel).

### MAJOR

**M1. Section 6.2 actively misrepresents what was run.**

WHERE: Section 6.2 (Backends), inset caveat starting "On the simulator and the live-LLM gap".

WHY: The caveat states the headline results "use the `sim` backend" and that the live-LLM run "was halted by an `insufficient_quota` error on the first generator call of every condition (every condition issued one request which returned 429 immediately)". Neither statement is true of the artifacts I inspected (`experiments/main_run_002_*`). The live-LLM run produced real gpt-4o-mini samples, completed all iterations, and is the actual source of the abstract's 0.58/0.40 numbers.

FIX: Remove or completely rewrite the caveat. Be explicit about which figure / table comes from which backend.

**M2. The internal data references in the prose do not match either run.**

WHERE: Section 7.3: "at iteration 2 `full_attrforge` sits slightly below `full_classic` in the per-iteration metric ($0.69$ vs $0.79$ on the same seed mean)".

WHY: I checked `experiments/sim_run_002_aggregated/per_iter.csv`. The simulator data shows iteration-2 macro F1: full_classic = 0.587, full_attrforge = 0.556 (not 0.79 vs 0.69). The live-LLM iteration-2 numbers (averaged across 3 seeds) are: full_classic mean (0.294 + 0.380 + 0.222)/3 = 0.299; full_attrforge (0.367 + 0.567 + 0.400)/3 = 0.445. The cited 0.69 vs 0.79 matches neither run.

FIX: Pull the actual numbers from `per_iter.csv` and report them honestly. If they support the narrative, great; if not, change the narrative.

**M3. The iter-1 dip rationalization in Figure 4 caption is a post hoc just-so explanation.**

WHERE: Figure 4 caption: "All conditions show a transient iter-1 dip, which we attribute to the simulator's generator responding to early prompt-updater instructions in ways that initially hurt label discriminability before consolidating."

WHY: In `per_iter.csv`, iter-1 F1 for self_critique, realism_only, diversity_only, and full_classic are ALL identical to 4 decimal places (0.3230). full_attrforge iter-1 is also exactly the same. This is consistent with the v1 review's "Bug B" (simulator produces the same samples regardless of which critics ran) only partially fixed. The "iter-1 dip" is not really a finding; it is a property of the simulator's prompt-content sensitivity being too weak to produce condition-distinguishing iter-1 samples. The live-LLM run does not show this consistent dip (it shows seed-dependent non-monotone trajectories per sanity_audit.md, and per the seed-17/23/41 numbers I checked).

FIX: Either drop Figure 4 (it's a simulator artifact, not a finding), or replace it with the live-LLM per-iteration curve. The simulator iter-1 dip says nothing about the loop's mechanism.

**M4. The Section 8 audit takeaways do not match the figure they describe in scale.**

WHERE: Section 8, Table 3 columns "Pack - null". `full_classic` is reported as -0.06 and `full_attrforge` as 0.00.

WHY: From `sim_run_002_aggregated/table.csv`: `pack_above_null_mean` for full_classic = -0.0625, full_attrforge = 0.0. Match. But Section 8 takeaway 1 then says "`full_attrforge` reaches the null reference exactly, which is the structural target". Reaching the null at 0.00 above is fine, but Section 8 then describes `diversity_only` and `full_classic` as "Pack accuracies below the null reference ($0.31$ and $0.44$ respectively)" and asks which interpretation that supports. The reader is left to reconcile: full_attrforge exactly hits the null target (good), but full_classic is below the null which the paper also calls "evidence against mode collapse" (good?). If two opposing outcomes are both good, the post-hoc audit is not actually distinguishing conditions. Either (a) the audit is differentiating them on real signal (in which case the paper should pick a side), or (b) the differences are within audit noise (in which case Table 3's claim of `full_attrforge` "landing exactly on the target" is over-reading random variation).

I also note that the audit Pack metric has zero standard deviation across 3 seeds (every entry says plus/minus 0.0000). With `n_comparisons = 16` and `pack_size = 4`, pack accuracy is quantized to 17 values; identical across 3 seeds suggests either the post-hoc audit consumed identical sample pools (which would imply the simulator produced identical samples per condition across seeds, see M3) or the seed propagation in posthoc_audit.py is not actually changing what the audit observes. Either way, the "audit decouples loop from observation" claim deserves a verification check.

FIX:
1. Add a paragraph explaining why "above null" and "below null" can both be "evidence against mode collapse" or otherwise reconcile them.
2. Inspect why audit pack accuracy SD is exactly 0 across 3 seeds and explain.

**M5. Table 1 ablation grid claim about "the only change is which critics are enabled" is partially false.**

WHERE: Section 6.3 caption: "the only change between rows is which critics are enabled".

WHY: I checked `baselines.py`. The `few_shot` baseline differs from `naive` in `enable_*` flags only trivially (label change) but ALSO bumps `generator.num_few_shot` from default to >=8. That's a generator config change, not just a critic-flag change. Furthermore, `naive` and `few_shot` use `iterations = 1`, while the others use `iterations = 3`. So those rows differ in (a) critic stack, (b) generator few-shot pool size (between naive and few_shot), AND (c) iteration count (between non-iterated and iterated rows), so the "only change" claim conflates 3 axes.

A subtler issue: the test set sample size for naive/few_shot is 16 synthetic samples vs 48 for iterated conditions. As the v1 review noted (E6), this confounds the comparison. The pivot framing does not fix it.

FIX: Either run naive/few_shot with the same 48-sample budget (3 separate single-iteration calls subsampled), or restrict the headline comparison to iterated conditions only and treat naive/few_shot as a separate "single-shot baseline" group.

**M6. The Mode-Seeking ratio "constant at ~0.23x" is presented as a negative finding, but the value 0.23 is not interpreted in any other run.**

WHERE: Section 7 Table 2 MS ratio (loop) column lists 0.22 +/- 0.01 for diversity_only and full_attrforge; Section 8 Table 3 audit gives 0.23 +/- 0.01 for every condition; Section 8 takeaway 2 calls this a negative finding.

WHY: I agree it's a negative finding for the simulator. But Section 7 then claims, almost adjacent to that text: "The full SynSmith stack moves the realism objective $0.03$ closer to chance than the three-critic baseline, while exactly matching it on attribute fidelity ($1.00$) and downstream metrics." So the only loop-internal advantage of full_attrforge claimed in Section 7.1 is a 0.03 shift on realism discriminator accuracy. With the noted 0.10 std on realism_only and 0.06 std on full_attrforge in the same table, that 0.03 shift is within one standard deviation, and the paper does not report a confidence interval on this difference either.

Also: the live-LLM realism numbers in sanity_audit.md show realism_only at 0.75 +/- 0.43, full_classic at 0.92 +/- 0.14, full_attrforge at 0.83 +/- 0.17. There the loop is moving discriminator accuracy AWAY from chance, not toward it. The paper's prose acknowledges this (Section 1 paragraph 3 and Section 9.1), but Table 2 still describes the simulator's 0.69 vs 0.72 simulator finding as positive. The paper is reporting two opposite-direction realism findings from two different backends in adjacent paragraphs without a clear reconciliation.

FIX: Report realism discriminator accuracy for both backends in a single table and tell the reader explicitly which is the source of which claim.

**M7. The "complaint-grammar contract" Section 4.4 Table 0 is not enforced.**

WHERE: Section 4.4: "structured named complaints admit locally targeted rewrites that a scalar reward does not".

WHY: Inspecting `synsmith/loop.py:444-516`, the updater is called with a single `IterationFeedback` object containing pack_artifacts, banned_phrasings, hole_exemplars, etc. as flat lists. The deterministic structure ends here; the actual prompt template the updater receives is built in `synsmith/updater.py` (not shown in my inspection above; I would expect to see it cited if the paper relied on it). The Section 4.4 "Table 0 is the core conceptual claim of our architecture" implies a specific grammar-of-rewrites table that is enforced. I see structured INPUTS to the updater, but no structured OUTPUTS or grammar of legal local rewrites. The Section 9.3 "Why named complaints beat scalar rewards" discussion is therefore qualitative.

FIX: Either (i) demonstrate via a controlled experiment that scalarizing the same critic outputs gives worse results, OR (ii) restate the claim more modestly as "we found a structure that worked; we did not test against scalarized alternatives in this paper".

**M8. The ref-gradcollide citation [28] is dated 2026 with arXiv ID 2605.26046, which is a future arXiv slot.**

WHERE: Section 2.2 and Section 9.6 cite this as documenting a 59% drop in gradient specificity, and frame it as motivating the multi-critic approach. References list: "Anonymous. *When gradients collide: Failure modes of multi-objective prompt optimization for LLM judges.* arXiv:2605.26046, 2026."

WHY: arXiv numbering is YYMM.NNNNN. 2605 corresponds to May 2026, beyond the model knowledge cutoff (January 2026) and beyond what should be in this submission (today's date is June 1, 2026 per the harness; so this would be a 1-month-old paper at best). The `related_work_scout.md` companion notes this: "#7 'When Gradients Collide' arXiv 2605.26046 is a future-dated ID (May 2026); cite as a current preprint." The paper cites it as if it is a published motivation for the multi-critic GAN-style transfer, including "directly motivates structural defenses like the ones we propose". If this paper does not exist or has a different content, then a load-bearing motivational citation is empty.

FIX: Verify the citation exists with a real URL (the arXiv link in the paper points at a slot that should exist if the ID is valid). If it is a real preprint, cite the verified author list (the current "Anonymous" is suspicious). If it is not real, remove the citation and find a verified anchor for the "naive multi-objective text-gradient methods degrade" claim.

**M9. The Section 9.4 claim that batch-level adversaries are "necessary" is rhetorical, not demonstrated.**

WHERE: Section 9.4 ("Why batch-level adversaries are necessary"): "Per-sample critics are blind to collapse for the same reason a vanilla GAN's per-sample discriminator is."

WHY: The supporting evidence in the paper would have to be: a condition where per-sample critics report "good" while batch-level adversaries detect collapse. The audit table (Table 3) almost provides this: `realism_only` (per-sample disc) reports 0.72 disc accuracy, but pack audit on its batch is 0.69 (well above null 0.50). Good, this is direction-correct evidence. But `full_classic` (still no batch-level adversaries) reports the same 0.72 disc accuracy and pack 0.44 (below null!). So the simulator's per-sample-only condition does NOT show the predicted "high disc, high pack" pattern uniformly. The "necessity" claim is not falsifiable as stated; "useful in some configurations" would be more accurate.

FIX: Reframe Section 9.4 as "Batch-level adversaries detect a different axis from per-sample critics" rather than "necessary".

**M10. The post-hoc audit "fresh seed" defense (posthoc_audit.py:105-108) re-instantiates critics per condition, but the audit Pack accuracy still has SD = 0 across 3 seeds in Table 3, which is suspicious.**

WHERE: Table 3 every row shows "0.XX plus/minus 0.00" for Pack accuracy.

WHY: With n_comparisons = 16 and pack_size = 4, the Pack accuracy is at best quantized to 17 values. Three seeds producing identical pack accuracies across all 7 conditions implies the random sampling of packs is deterministic across seeds, which is intentional per the audit's "RNG re-seeded per condition" design (line 117-118: pack_cfg = PackDiscriminatorConfig(pack_size=4, n_comparisons=16, seed=99)). The seed 99 is hard-coded for the audit, so the audit explicitly does NOT seed-randomize across the 3 seed runs. The "mean +/- std over 3 random seeds" caption is misleading: the seeds vary the loop, but the audit pack discriminator gets the same seed every time. If the simulator + loop produce identical samples per condition across seeds (which is what M3 hints at), then the audit pack accuracy will be identical.

FIX: Either (i) randomize the audit seed across seed runs, document it, and report real std (which will be nonzero), or (ii) be explicit in the figure caption that the audit pack RNG is fixed and the "SD = 0" reflects only the simulator's output identity across the 3 outer seeds.

**M11. The "78% of architectures stronger" framing in Section 1 ("by every architectural measure stronger than its three-critic ablation") is not falsifiable.**

WHERE: Section 1 paragraph 2: "The resulting seven-critic loop is by every architectural measure stronger than its three-critic ablation."

WHY: "Architectural measure" is undefined. If it means "more critics, longer prompts, more constraints", then trivially yes. If it means "every numerical loop-internal metric is at least as good for full_attrforge", then it is also trivially true for some metrics (more constraints in the prompt) but false for others (the simulator's near-duplicate rate is 2x higher for full_attrforge than full_classic).

FIX: Either drop the phrase or replace with a specific list of measures and their numbers.

**M12. The "12 LLM-tic phrasings" finding is also seed-fragile.**

WHERE: Abstract, Section 1, Section 7 prose.

WHY: I checked the per-seed banned_phrasings_total: seed 17 = 12, seed 23 = 11, seed 41 = 12 (live LLM). Mean is 11.67. Picking "12" as the headline is rounding from 11.67 in a way that suggests stable measurement. With min_repeats = 1 (per BL3), this is also essentially "the LLM emitted 12 distinct things the Mode Hunter LLM happened to flag".

FIX: Report mean +/- std on the banned_phrasings_total across seeds.

**M13. The "Reproduce with" CLI invocation in Section 6.3 points at `--config examples/customer_support/config.sim.yaml`, but the artifact-producing run for the live-LLM headline uses the openai backend.**

WHY: The CLI shown produces the simulator (sim_run_002), not the live LLM run (main_run_002). A reader following the CLI literally would reproduce the 0.76/0.76 result, not the 0.58/0.40 result. There is also no `examples/customer_support/config.sim.yaml` referenced in the artifact tree, that I could find; the closest is `examples/customer_support/config.yaml` which uses the openai backend. The CLI may be silently broken.

FIX: Verify the CLI runs cleanly. Provide separate reproduce-commands for the simulator and live-LLM runs.

### MINOR

**N1.** "research preview v0.5" header label is fine, but the bibliographic year "2026" is set; the actual evidence base is 3 seeds, 1 task. TMLR is fine with research-preview framing but the abstract should be consistent.

**N2.** Section 2.4b cites "NanoFlux" as "the closest existing GAN-style framework for LLM data". The companion `related_work_scout.md` notes that author lists for ref-nanoflux, ref-paretoprompt, ref-debatejudges, ref-gradcollide, and ref-strongcollapse "were not directly verified (search-result based); verify with WebFetch on each before final submission." The current bibliography still lists "Anonymous" for these, which suggests the verification did not happen. This is a soft red flag for the comprehensiveness of the related-work scout.

**N3.** Table 0 row "Mode-Seeking" has the example complaint "changing style does not change text; per-attribute sensitivity[style] = 0". I checked: the per-attribute sensitivity matrix IS computed in `mode_seeking.py:127-141`, but its values are populated as `attribute_sensitivity` in the `ModeSeekingResult`. I cannot easily confirm whether the prompt updater consumes the per-attribute breakdown vs only the scalar; the v1 review (G6) noted "attribute_sensitivity is populated but never consumed by the simulator updater". Worth confirming.

**N4.** The hero figure (figures/hero.png) is reused as Figure 1; the v1 review (M11) asked for an actual architecture box-and-arrow diagram. The "figures/architecture.png" referenced by Figure 1 likely now is that, but the alt text suggests it shows the same conceptual layout repeatedly. Worth checking that Figure 1 actually contributes new information vs the hero image.

**N5.** Section 9.2 ("What the validation does not show") helpfully acknowledges the 10-test-item ceiling, but then says "all iterated conditions saturate at the same F1, which we attribute to the noise floor of a 10-item test set". This claim is only true for the simulator run; the live-LLM run does NOT saturate (the 0.58 vs 0.40 gap is precisely the point). The reframing is internally inconsistent.

**N6.** Section 9.3 ("Why named complaints beat scalar rewards") gives a qualitative argument; no controlled experiment compares scalar-aggregated critic feedback against named-complaint feedback. The "We argue" framing should be made explicit.

**N7.** The bib_validation.md companion shows that ref-progen was previously wrong (wrong authors, wrong venue). I checked the corrected entry in docs/index.html: it now reads "Ye, J., Gao, J., Wu, Z., Feng, J., Yu, T., & Kong, L." and "Findings of EMNLP, 2022", matching bib_validation.md's correction. Good. But ref-inpars and ref-textgrad still carry the cosmetic title variants that bib_validation flagged.

**N8.** The "On the simulator and the live-LLM gap" caveat (Section 6.2) uses the phrase "an `harness validator`" (typo: should be "a harness validator", since "harness" starts with a consonant sound).

**N9.** Section 5.1 PacGAN derivation: the Pack Discriminator uses M random pair comparisons of a pack of k real and a pack of k synthetic. The original PacGAN concatenates k samples into one input to the discriminator (still per-pair classification but k-channel input). The paper's framing as "the judge is asked which pack is the LLM pack" is closer to "two-sample test" than to PacGAN per se; mention this so a careful reviewer doesn't bounce on the analogy.

**N10.** The "Why a prompt, not weights?" callout in Section 3 says "no GPU required" as advantage (iv). This is misleading for the live-LLM run, which still calls an LLM API for every iteration. The "no GPU on our side" is what's meant. Clarify.

**N11.** Section 5.3 says "A cap of 50 entries bounds prompt bloat" and the code (`ModeHunterConfig.max_banned_total = 50`) confirms this. With min_repeats = 1 from BL3, the cap is reachable in 13 iterations at max_findings_per_iter = 4; not a tight bound for short runs but worth noting if longer runs are tried.

**N12.** Figure 5 ("per-class F1") is labeled "sim_run_001" in the filename (`sim_run_001_per_class_f1.png`), one major run earlier than the headline run. Either re-render from sim_run_002 or remove.

## Empirical concerns

The deepest empirical worry is that the central claim ("a diversity-discriminability tradeoff", 0.58 vs 0.40) depends on:
1. A single low-capacity downstream classifier (TF-IDF + LR);
2. A test set of 10 items, 2 per class;
3. Three seeds, with the gap dominated by 2 of them (paired diffs 0.23, 0.27, 0.07);
4. A live-LLM run for which the manuscript does not provide an aggregated table or audit (the post-hoc audit results in the paper come from the simulator, which separately shows no F1 gap between full_classic and full_attrforge).

In this configuration, the right interpretation of the gap is "TF-IDF + LR loses keyword consistency when the prompt loop injects surface variation, on this specific 10-item test, on 3 seeds". That is a real and interesting observation. But it does NOT support the "more critics, worse data" framing the paper wants, which is a much broader claim about synthetic data quality. The chain from "TF-IDF F1 drops" to "the data is worse" is not established here. The paper acknowledges this in spots (Section 9.2) but the abstract and Section 1 do not.

Additionally:
- The Pack-accuracy reported in the loop (full_attrforge = 0.50, Table 2) is structurally tautological with the audit (full_attrforge tries to drive Pack to 0.50); the post-hoc audit using a different seed for the Pack discriminator finds full_attrforge = 0.50 ± 0.00. The fact that the audit replicates exactly the loop target value is consistent with "the audit's Pack RNG is the same fixed seed across the 3 seeds" (M10).
- The "above null" framing is potentially circular: the null is real-vs-real pack, but if the same Pack discriminator is biased toward labeling certain real subsets as "synthetic", the null itself is not a clean baseline. The paper mentions this once (Section 5.1) but the "exactly on the target" celebration in Section 8 is uncareful.
- Attribute match below 10% is reported as a finding but not explained mechanistically. The paper hints (Section 1 paragraph 3) but never analyzes why the verifier's match rate is so low. With the simulator's verifier having a rigid "difficulty hard ⇒ keyword_hits >= 1 AND len < 80" rule (v1 review G7), one suspects the simulator's verifier is also fighting itself; the live-LLM verifier may be different.

## Methodological gaps (claim vs code)

- **MG1**. Section 5.3 paper rule "appear in >= 2 synthetic samples" vs config `min_repeats: 1` (see BL3 above).
- **MG2**. Section 4.4 "rewrite is constrained by a length budget (280 words)". The Limitations section (item 8) admits "The 280-word budget is rendered as an instruction in the user message but not enforced programmatically by the simulator updater; it falls back to a soft 8-clause cap with deduplication." OK; the limitation is acknowledged, but Section 4.4's main text still asserts the budget as a hard constraint of the method. Reconcile.
- **MG3**. Section 5.2 promises "we additionally report a per-attribute sensitivity matrix" and "values << 1.0 indicate attribute-deaf generation". The matrix is computed (`mode_seeking.py:127-141`) and emitted as `attribute_sensitivity`. But the matrix is not shown anywhere in the paper. The paper says the matrix exists but does not render it.
- **MG4**. Section 7.4 references `figures/sim_run_001_per_class_f1.png`, a figure from a prior run, while the rest of Section 7 references sim_run_002. Stale figure.
- **MG5**. Coverage Hole Finder's classifier is described as a "logistic regression on TF-IDF features" (Section 5.4). The code matches (`coverage_hole.py:78`). But the classifier_auroc = 1.00 means the classifier perfectly separates real from synthetic on every condition. The paper notes this as a negative finding. The cause is almost certainly the low-data regime: 30 real + 16-48 synthetic, with a fully-flexible logistic regression and any nontrivial vocabulary overlap, AUROC -> 1.0 is the expected high-variance failure mode of LR on small N. The paper does not interrogate whether this is an intrinsic coverage failure or a small-N AUROC artifact. With N this small, AUROC = 1.0 may be uninformative; a regularized or N-bootstrap variant would be more honest.

## Writing and clarity issues

- **W1**. The two-narrative gluing (live-LLM in the abstract, simulator in the tables and figures) confused this reviewer for several minutes. A single decision (live OR sim as the headline) with the other clearly relegated to an appendix/sanity-check section would help enormously.
- **W2**. The "diversity-discriminability tradeoff" name is catchy but the paper does not formalize either axis. "Diversity" is variously: pack-acc-toward-null, mode-seeking ratio, near-duplicate rate, banned-phrasings count, plus "surface diversity, by any direct measurement". "Discriminability" is TF-IDF macro F1 only. The asymmetry weakens the conceptual claim.
- **W3**. Section 1 paragraph 2 uses bold "0.58 plus/minus 0.05" and "0.40 plus/minus 0.06" right next to the unbolded "across 3 random seeds". Boldface invites the reader to skip the seeds detail. Reverse the emphasis or attach the statistical reservation to the bold numbers.
- **W4**. The italic "Adding more critics actively hurts downstream classifier performance." is a stronger claim than the data supports under N=3 paired and N_test=10. Soften.
- **W5**. The "honest negative findings" framing is fine and a strength, but the cluster (mode-seeking constant, AUROC at 1.0, attribute-match below 10%, realism drifting away from chance) is more than three findings. Either consolidate or list them all.
- **W6**. "More critics, worse data?" question-form title is rhetorically punchy but the paper does NOT show "worse data" in any sense other than "harder for one specific weak classifier". Title should at minimum read "More Critics, Harder Downstream for a Weak Classifier?" or similar.
- **W7**. The "complaint grammar contract" is named in the abstract ("a complaint-grammar contract that lets future work add new critics") but never crisply defined as a formal interface. If it is a major contribution, write it as such (e.g., a table of types: `Critic.feedback: {complaints: List[Named, Reasoned]}`, etc.).
- **W8**. Section 8 takeaway 1 says "All iterated conditions sit at 0.77 / 0.76 on downstream metrics (Table 2)". Reading flow: Table 2 reports per-condition numbers; the prose conflates accuracy and macro F1 in "0.77 / 0.76". This is a minor wart but adds to W1.

## Missing related work

The related-work scout found 13 candidates and the paper added most of them. The principal remaining gap is on the downstream-classifier-choice axis: the entire mechanism the paper proposes hinges on "TF-IDF + LR loses keyword consistency". The literature on "downstream metric sensitivity to synthetic data" includes:
- Liu et al., G-Eval (already cited): on the LLM-judge end.
- The "Synthetic Eggs in Many Baskets" paper (cited): empirical anchor for the TSTR protocol.
- Missing: any paper that decouples synthetic-data quality from downstream-classifier capacity. A useful one is van Breugel & van der Schaar's 2023 *Synthetic Data, Real Errors* (ICML 2023, "Beyond Privacy: Navigating the Opportunities and Challenges of Synthetic Data"), which explicitly stresses that downstream-classifier choice is a confound.
- Missing: any paper that runs TSTR (train-on-synthetic, test-on-real) with multiple downstream classifiers and reports which-classifier-shows-what. This is the closest related-work category for BL5.

## Overall recommendation

**MAJOR REVISIONS.**

The reason is not that the paper's pivot is bad; the diversity-discriminability framing is intellectually interesting and the negative-result honesty is creditable. The reason is that the manuscript as written:
1. Contains a glaring internal contradiction (abstract live-LLM numbers vs Table 2 simulator numbers) that any TMLR reviewer will notice immediately;
2. Cites a future-dated paper (BL8 / M8) as load-bearing motivation;
3. Reports statistical claims that the data, on close inspection, do not support (BL2);
4. Misrepresents the Mode Hunter rule (BL3);
5. Has not addressed the central confound of its main finding (BL5).

Each individually is fixable. Together they require a substantial rewrite plus an additional experiment (downstream-classifier sweep). TMLR's "claims vs evidence" rubric is precisely targeted at this kind of paper, where the engineering and the framing are both legitimate but the evidence currently underwrites a narrower claim than the manuscript states.

I would recommend the authors not weaken the contribution; they should instead deepen it. The "TF-IDF + LR loses keyword consistency" finding, if confirmed against 2+ stronger classifiers AND reframed as a quantitatively measured surface-diversity vs label-keyword-consistency tradeoff, would be a robust, citable contribution.

## Top 7 fixes the authors should make before resubmission

1. **Resolve the abstract-vs-Table-2 contradiction (BL1).** Pick one backend as the headline. If live-LLM is the headline, regenerate Figures 2/3/4/6/7 from `main_run_002`. Provide an `experiments/main_run_002_aggregated/` directory analogous to `sim_run_002_aggregated/`.

2. **Replace the "factor of three" statistical framing with a real significance test (BL2).** Report paired t and Wilcoxon p-values, paired bootstrap CIs on the mean difference. Acknowledge that p = 0.09 on N=3 paired with the highly-variable per-seed gap (seed 41 = 0.07; seeds 17 and 23 = 0.23 to 0.27) makes the gap suggestive but not conclusive at this N. Either run more seeds, or temper the abstract.

3. **Run the downstream evaluation with at least two stronger classifiers (BL5).** Sentence-transformer + LR; gpt-4o-mini zero-shot intent classifier. If the gap persists, the central finding strengthens dramatically. If it inverts or vanishes, restate the finding as "TF-IDF + LR is sensitive to surface variation in synthetic data".

4. **Fix the Mode Hunter rule misrepresentation (BL3).** Either restore `min_repeats = 2` and rerun (the headline 12 will likely change), or rewrite Section 5.3 to honestly state `min_repeats = 1` was used and explain why.

5. **Verify or remove the future-dated citation (M8/N2/BL).** ref-gradcollide arXiv:2605.26046 needs a real author list and a working URL. Same for the other "Anonymous" citations the related-work scout flagged.

6. **Add a direct-diversity-measurement table for the live-LLM batches (BL4).** Compute Distinct-n, Self-BLEU, and embedding-cluster count for `full_classic` vs `full_attrforge` final batches, across all 3 seeds. Report mean +/- std. If "surface diversity measurably increases", these three metrics should all increase. If they do not, weaken the claim.

7. **Rewrite Section 6.2 to honestly describe what was run (M1).** Drop the "halted by `insufficient_quota`" caveat (which does not match the artifacts), and add a clear sentence: "Two complete runs are reported: `sim_run_002` (deterministic simulator, 3 seeds), provides reproducibility and audit-protocol demonstration; `main_run_002` (gpt-4o-mini, 3 seeds), provides the headline downstream-F1 finding."

## Confidential comments to the action editor

This paper is a genuine borderline case. The engineering is solid; the GAN-style adversaries are a legitimate transfer; the post-hoc audit protocol is a real methodological improvement over v1; and the authors' willingness to pivot the framing toward an "honest negative finding" rather than spin the data is professionally creditable.

However, the manuscript is, in its current form, internally inconsistent in ways that any careful reviewer will catch within 30 minutes (the simulator-vs-live-LLM gluing is the most glaring). The headline statistic (0.58 vs 0.40 with N=3 paired, p = 0.09 on a 10-item test) is, charitably, "suggestive evidence in the direction the authors describe", not "the gap is 3x the seed std" as the abstract claims. The central mechanism claim ("diversity injection breaks classifier keyword consistency") is plausible but is gated entirely on a single low-capacity classifier choice that the authors have not stress-tested.

I think the right disposition is "major revisions with a clear path to acceptance". The required additional work is bounded:
- One re-aggregation pass over the existing live-LLM artifacts;
- One additional experiment running 2 stronger downstream classifiers;
- One config fix to `min_repeats` or one prose fix to Section 5.3;
- One citation cleanup;
- A substantive rewrite of Sections 6.2, 7, and the abstract.

If the authors execute these well, the contribution becomes a robust "named-complaint-driven prompt loop introduces measurable surface diversity at a measurable cost to keyword-driven downstream classifiers" finding. That is a real and useful claim. If they do not execute, the paper will be reasonably rejected at the next round.

My read on the authors: the companion sanity_audit.md is a strong indicator that they ARE aware of most of these issues internally. The bib_validation.md and related_work_scout.md are evidence of methodical revision discipline. The pivot from "SynSmith improves" to "diversity-discriminability tradeoff" is the right direction. The issue is that the manuscript revision did not catch up with the analysis discipline elsewhere in the repo. With one more careful pass, this paper can be acceptable.

I would not be surprised if a different reviewer recommends outright reject on BL1 alone (the abstract/Table-2 contradiction is severe). I think that would be unfair given the underlying work; major revisions is fairer.
