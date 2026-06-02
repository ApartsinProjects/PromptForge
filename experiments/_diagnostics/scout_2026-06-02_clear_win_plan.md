# Scout: how to push AttrForge from tied contributor to clear winner

Date: 2026-06-02
Source: web-researcher subagent

## Top-3 pursue-first experiments (combined budget ~$50, ~1 day wall-clock)

### 1. Switch headline benchmark to Synthline 4-task RE suite + report DCScore alongside macro F1

**Why highest leverage:** Saturated 10-item customer-support test is the bottleneck. Synthline paper (El-Hajjami & Salinesi 2025, [arXiv:2506.21138](https://arxiv.org/abs/2506.21138), EMNLP 2025) shows multi-critic loops produce 6-44 F1pt lifts on RE classification. DCScore ([arXiv:2502.08512](https://arxiv.org/abs/2502.08512), ICML 2025, [GitHub](https://github.com/bluewhalelab/dcscore)) is the discriminative diversity metric that distinguishes "useful iteration" from "more compute". distinct-n misses this.

**Plan:** Run all 7 critic configs on the 4 Synthline RE tasks. ~$15-30 OpenAI Batch, ~6h wall-clock. If AttrForge wins on RE, it's a clear win; if it ties, that's itself a finding.

### 2. Replace deterministic GAN adversaries with one co-trained GAD-style realism discriminator

**Why second highest:** "AttrForge solo loses to full_classic" is because the four GAN heads are deterministic and cannot adapt to generator drift. GAD ([arXiv:2511.10643](https://arxiv.org/html/2511.10643v1)) shows minimax co-training works in black-box LLM distillation: initialize from a frozen DeBERTa-v3-base, update 1 epoch per critic-loop iteration on (real, AttrForge-synthetic) pairs.

**Plan:** ~$2-5 on Modal A10G, 2-4h wall-clock per dataset. Replaces the 4 deterministic adversaries with 1 learned co-updating discriminator.

### 3. Adopt paired BCa-bootstrap + sign-flip protocol from "+1% is not enough" ([arXiv:2511.19794](https://arxiv.org/html/2511.19794v1))

**Why third:** Either flips currently-NS results to significant under more powerful pairing, or definitively confirms they're noise. Either outcome resolves the "p=0.06 at N=10" hand-wringing. No new experiments needed; pure re-analysis.

**Plan:** ~$0, ~2h analyst-time.

## Full 5-category scout output

### A. Datasets where multi-critic loops PROVABLY win
- **Synthline RE suite** (arXiv:2506.21138): 6-44 F1pt lifts; AttrForge plugs in as critic ensemble.
- **SIPDO closed-loop QA/reasoning** (arXiv:2505.19514): generator-finds-weakness + optimizer-fixes-prompt > standard prompt tuning. AttrForge's banned-list + Coverage Hole map onto SIPDO's reveal-weakness hook.
- **Low-resource multilingual classification** (arXiv:2601.16278, 2026): LLM-as-generator + critic ensemble beats LLM-as-classifier most strongly when labels are scarce.

### B. Strengthen the GAN-style adversarial side
- **Generative Adversarial Distillation / GAD** (arXiv:2511.10643): minimax co-training in black-box LLM distillation.
- **SyNeg hard-negative synthesis** (arXiv:2412.17250): use Coverage Hole module to generate semantically-close-but-wrong-intent hard negatives; train realism critic with InfoNCE on them.
- **Self-Adversarial Comparative Discrimination** (arXiv:2001.11691): pairwise "better than last sample?" discriminator; attacks GAN-on-text reward sparsity.

### C. Better metrics that surface multi-critic wins
- **DCScore** (arXiv:2502.08512, ICML 2025): classification-based diversity; beats Self-BLEU/Distinct-n in correlation with downstream gain. Drop-in replacement for distinct-n.
- **Simulated Annotators** (ICLR 2025): reduces ECE by ~50%, AUROC +13% on GPT-4. Report ECE/Brier alongside macro F1.
- **RewardBench pairwise discriminative protocol**: score critic ensembles head-to-head (sc+full_attrforge vs sc+diversity_only) on a pairwise discriminative test.

### D. Method modifications for 2x+ realism/diversity gains
- **Activation Steering** (arXiv:2605.28664, 2026): harmonic mean of success-coherence-diversity is the right tuning target.
- **Schema-based persona steering** (arXiv:2509.15447, Sept 2025): significantly reduces repetition vs free-form personas.
- **TextGrad** (Nature 2025): textual gradients from a judge's NL critique mutate the system prompt. Plug AttrForge's 3 LLM critics in as TextGrad's TextLoss.

### E. Statistical methods that flip NS to significant
- **Paired BCa bootstrap + sign-flip permutation** (arXiv:2511.19794, Nov 2025): designed for small-seed budgets; discriminates 0.6-2.0pp gains correctly at modest seed counts.
- **Linear mixed-effects bootstrap** (Lohse 2022, arXiv:2207.12455): seed as random intercept, condition as fixed effect; 30-50% effective power vs paired-seed t-test at N=10.
- **Stratified bootstrap by per-class worst-F1**: bootstrap WITHIN the worst-performing class rather than over the whole macro; directly tests the "+0.233 worst-class lift" claim.

## Gaps the scout flagged

- El-Hajjami & Salinesi PDF stream not parseable via WebFetch; the 6-44 pp range is from the arXiv abstract.
- GAD discriminator architecture details (2511.10643) not verified.
- No 2024-2026 result evaluates a multi-critic loop on Banking77 with a positive lift; this is publication-worthy if AttrForge wins there.
