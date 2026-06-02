# v2 plan: harder cases + stricter metrics

The v1 paper had one statistically significant downstream finding (the TF-IDF
isolated gap at p=0.046, which is a *cost* for SynSmith, not a win) plus a
robust-but-NS variance-reduction observation on the `complaint` class. The
binding constraint is that the customer-support task is too easy: three of
five classes already saturate at F1=1.00 from real data alone, and macro F1
averages the easy classes with the hard one, hiding any signal.

v2 plan: surface harder cases and stricter metrics that the diversity story
should win on. Three packages, executed in order.

## Package B (in progress): same data, stricter metrics

Free experiments on the existing synthetic batches. Targets:

1. **Worst-class F1** across all conditions × all real-train sizes. The
   `complaint` class never saturates at the real-only ceiling and is where
   SynSmith already shows ±0.20 directional advantage. Going from macro to
   worst-class should turn the v1 NS observation into a significant one.

2. **Adversarial test-set robustness**: paraphrase the 10-item real test set
   via back-translation or LLM-paraphrase, then re-evaluate the augmented
   classifiers from each condition. SynSmith's higher lexical diversity
   should produce classifiers that degrade *less* under paraphrase. ~$0.10
   in API calls for the paraphrase pass; the downstream evaluation is CPU.

3. **Expected Calibration Error (ECE) and Brier score** on the augmentation
   predictions. Better-calibrated downstream classifiers are a different
   axis on which diversity can win.

4. **Sample-efficient ceiling**: at what synthetic-sample budget does each
   method reach 0.85 F1? Plot the sweep.

## Package A: harder benchmark (next, ~$50, 2 days)

Switch primary task to a **Banking77 10-class subset** of overlapping
intents (e.g. `card_lost`, `card_blocked`, `cancel_transfer`,
`declined_card_payment`, etc.). Same 7 conditions, same 5 seeds, same
augmentation protocol. Test set 50+ items per class.

## Package C: regime extension (next, ~$10)

Re-run on the existing task at n=0 (pure synthetic recovery) and n=1 per
class. Compositional generalization: hold out one class entirely.

## Branch policy

- `v2` is the new working branch
- `main` stays as the v1 published version
- Merge to `main` when v2's empirical story is consolidated and reviewed
