# Related-work scout: recent (2024-2026) papers to consider

13 candidate citations + 4 optional. Compiled by independent web research
against arXiv, OpenReview, ICLR proceedings.

| # | Citation | Year / Venue | Category | One-line summary | Why cite | URL |
|---|---|---|---|---|---|---|
| 1 | Zhang et al., **Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity** | arXiv 2510.01171, Oct 2025 | LLM mode collapse | Training-free prompting that asks the model to verbalize a probability distribution over responses; 1.6-2.1x diversity gain in creative writing. | Direct prior art for the mode-collapse-defense framing. | https://arxiv.org/abs/2510.01171 |
| 2 | **Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning** | arXiv 2511.01490, Nov 2025 (Findings of ACL 2026) | Train-on-synthetic | Higher source diversity in synthetic training data measurably mitigates distribution collapse during LLM fine-tuning. | Empirical anchor for the TSTR protocol; cite in model-collapse discussion. | https://arxiv.org/abs/2511.01490 |
| 3 | **Strong Model Collapse** | ICLR 2025 | Model collapse | Proves model collapse persists even with arbitrarily small synthetic fractions in mixed real+synthetic training. | Rigorous follow-up to Shumailov 2024; justifies defenses. | https://proceedings.iclr.cc/paper_files/paper/2025/file/284afdc2309f9667d2d4fb9290235b0c-Paper-Conference.pdf |
| 4 | Kazdan et al., **A Note on Shumailov et al. (2024)** | arXiv 2410.12954, Oct 2024 | Model collapse follow-up | Shows collapse is sensitive to data-mixing strategy; accumulating real data prevents it. | Honest counterweight to one-sided model-collapse framing. | https://arxiv.org/abs/2410.12954 |
| 5 | **Machine-generated text detection prevents language model collapse** | arXiv 2502.15654, Feb 2025 | Synthetic data detection | MGT detector as a filter in the data pipeline prevents collapse across generations. | Precedent for "filter-by-classifier" thread; aligns with our density-ratio filtering. | https://arxiv.org/abs/2502.15654 |
| 6 | Li et al., **Auto-Prompt Ensemble for LLM Judge** | arXiv 2510.06538, Oct 2025 | Multi-critic prompt opt | Confidence-aware ensemble of multiple prompt candidates for LLM-as-judge; +3.3 pp on Reward Bench. | Direct competitor to multi-critic update loop; contrast with our minibatch defenses. | https://arxiv.org/abs/2510.06538 |
| 7 | **When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges** | arXiv 2605.26046, 2026 | Multi-critic prompt opt | Gradient specificity in TextGrad/OPRO/GEPA drops 59% when multiple criteria are jointly optimized. | Motivates GAN-style minibatch defenses because naive gradient aggregation provably degrades. | https://arxiv.org/abs/2605.26046 |
| 8 | Hu et al., **Multi-Agent Debate for LLM Judges with Adaptive Stability Detection** | arXiv 2510.12697, Oct 2025 | Multi-critic / debate | Formal proof that debate amplifies judge correctness vs static ensembles. | Orthogonal to our adversarial dynamics; cite in "ensemble of critics". | https://arxiv.org/abs/2510.12697 |
| 9 | **Pareto Prompt Optimization** | ICLR 2025 | Multi-objective prompt opt | RL method using dominance relations to explore the full Pareto front of prompts. | Important comparator for the multi-objective claim. | https://openreview.net/forum?id=HGCk5aaSvE |
| 10 | Agarwal et al., **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** | arXiv 2507.19457, Jul 2025 | Prompt opt baseline | Reflection-based prompt evolution with Pareto-front selection outperforming RL. | Primary 2025 SOTA prompt-optimizer baseline. | https://arxiv.org/pdf/2507.19457 |
| 11 | **NanoFlux: Adversarial Dual-LLM Evaluation and Distillation for Multi-Domain Reasoning** | arXiv 2509.23252, Sep 2025 | GAN-style adversaries on LLM outputs | Attacker/Defender LLM pair supervised by a tool-augmented Judge. | Closest existing GAN-style framework for LLM data generation; mandatory citation. | https://arxiv.org/pdf/2509.23252 |
| 12 | **Improved Density Ratio Estimation for Evaluating Synthetic Data Quality** | ICLR 2025 SynthData workshop | DRE for synthetic data | Aggregates multiple DRE models for global+local utility scores. | Direct technical antecedent for binary-classifier filtering. | https://openreview.net/forum?id=IBnCO8gtLA |
| 13 | Jandaghi et al., **Scendi Score: Prompt-Aware Diversity Evaluation via Schur Complement of CLIP Embeddings** | arXiv 2412.18645, Dec 2024 | Embedding diversity | Decomposes embedding-based diversity into prompt-driven vs intrinsic-model components. | Closest 2024 embedding-based diversity alternative beyond distinct-n / FID. | https://arxiv.org/pdf/2412.18645 |

## Optional (mention only if space)

| 14 | Tournament of Prompts | arXiv 2506.00178 | https://arxiv.org/html/2506.00178v1 |
| 15 | MOPrompt | arXiv 2508.01541 | https://arxiv.org/html/2508.01541v1 |
| 16 | Embedding-Driven Diversity Sampling | arXiv 2501.11199 | https://arxiv.org/abs/2501.11199 |
| 17 | Curse of Recursion | arXiv 2404.05090 | https://arxiv.org/pdf/2404.05090 |

## Caveats from the scout
- Category 2 (GAN-style adversaries against LLM batches) is thinly populated. NanoFlux (#11) is the closest match; no other 2025 paper directly frames a multi-critic prompt loop as PacGAN/MSGAN for LLM outputs. **This gap is part of our contribution narrative.**
- Author lists for #3, #6, #7, #11 were not directly verified (search-result based); verify with WebFetch on each before final submission.
- #7 "When Gradients Collide" arXiv 2605.26046 is a future-dated ID (May 2026); cite as a current preprint.
- No clean Sep 2024-2026 paper on "LLM-stylometry-as-mode-collapse-detector" was surfaced.
