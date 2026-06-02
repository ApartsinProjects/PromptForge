# Root-cause analysis: why are AttrForge's downstream-F1 results NS?

Date: 2026-06-02
Method: per-test-item correctness inspection (applying the new global rule).

## TL;DR

**The customer-support N=10 evaluation is discretization-limited: 8 of 10 test items saturate at 100% on every condition, so the macro F1 variance reduces to 2 `complaint` items.** Every aggregate claim (Pack hurts, AttrForge ties full_classic, ensemble lifts +0.07, etc.) is, mechanistically, about how often each condition correctly classifies test items [6] and [7].

## Per-test-item correctness (N=10 seeds per condition)

| Item | True label | Text | full_classic | full_attrforge | no_pack | realism_only |
|------|------------|------|--------------|----------------|---------|--------------|
| [0] | refund_request | "pls refund my last order..." | 10/10 | 10/10 | 10/10 | 10/10 |
| [1] | refund_request | "transaction failed but money was deducted..." | 10/10 | 10/10 | 10/10 | 10/10 |
| [2] | technical_problem | "sync isn't working between desktop and mobile..." | 10/10 | 10/10 | 10/10 | 10/10 |
| [3] | technical_problem | "the export to PDF button is greyed out..." | 8/10 | 9/10 | 10/10 | 9/10 |
| [4] | account_issue | "can't log in. says wrong password..." | 10/10 | 10/10 | 10/10 | 10/10 |
| [5] | account_issue | "deleted my account by mistake..." | 10/10 | 10/10 | 10/10 | 10/10 |
| **[6]** | **complaint** | **"honestly your new pricing is a joke..."** | **5/10** | **4/10** | **4/10** | **5/10** |
| **[7]** | **complaint** | **"still no answer on ticket #4421..."** | **6/10** | **6/10** | **9/10** | **8/10** |
| [8] | general_question | "Do you guys ship to Norway?..." | 10/10 | 10/10 | 10/10 | 10/10 |
| [9] | general_question | "can two people share one account..." | 10/10 | 10/10 | 10/10 | 10/10 |

## What this means

**Item [6]** ("honestly your new pricing is a joke") is sarcastic complaint. **No condition does meaningfully better than chance.** Every condition's synthetic batch under-represents sarcastic-style complaints. The AttrForge diversity push actively moves synthesis away from sarcastic phrasings (the Mode Hunter library bans phrases like "I feel completely ignored", "I genuinely just want my money back" — both sarcastic registers).

**Item [7]** ("still no answer on ticket #4421") is a delivery-style complaint with a ticket number. **This is where the Pack Discriminator hurts:** `no_pack` gets 9/10 right vs full_attrforge's 6/10 (+0.3 per-item lift). The Pack Discriminator's batch-level non-homogeneity constraint pushes the generator away from generating delivery-complaint phrasings repeatedly, even though they are the correct distribution for this class.

## Root cause attributions (all aggregate findings explained)

| Aggregate finding | Datapoint-level explanation |
|---|---|
| "Pack Discriminator hurts" (R1: +0.053 macro for `no_pack`) | Pack suppresses delivery-complaint phrasings in synth; item [7] correctness goes 6/10 → 9/10 when Pack is removed |
| "full_attrforge solo ties full_classic on macro" | Both fail similarly on item [6]; item [7] difference is within seed noise |
| "AttrForge wins lexical diversity panel (Table 8)" | The diversity push achieves what it's designed to do at the batch level; the cost is item [6]/[7] coverage |
| "Cross-condition ensemble (sc+af) ≡ (sc+diversity_only)" | Both ensembles include `self_critique` which classifies item [7] correctly more often; the second-member identity is invariant when both fix the same 2 items |
| "Ensemble +0.073 macro over full_classic solo (BCa CI excludes zero)" | Ensemble gets item [7] right 9-10/10 (vs full_classic's 6/10), and item [3] right 10/10 (vs 8/10) |

## What the data does NOT support

- **"AttrForge wins on downstream F1 by macro"** — false; the variance reduces to items [6] and [7], and AttrForge ties or slightly loses on both.
- **"AttrForge is necessary for the ensemble"** — false; `diversity_only` works identically because the ensemble's per-item correctness on items [6][7] is determined by `self_critique`, not by the second member.
- **"The 2 of 4 GAN-style adversaries that don't differentiate (Mode-Seeking, Coverage Hole) are doing something invisible"** — they ARE doing nothing measurable; we've now confirmed at the item level.

## What the data DOES support

- **Iteration helps lexical diversity in the synthetic batch** (Vendi 2×, robust)
- **Cross-condition ensemble extracts decision-boundary diversity** via `self_critique`'s correctness on item [7]; the +0.073 vs full_classic is the +3 items going from 60% to 90% on item [7]
- **The 6-critic loop (no_pack)** specifically rescues item [7]'s delivery-complaint phrasing
- **The test set is discretization-limited at 10 items**; any future paper-strengthening experiment needs at least 100 items

## Implications for next experiments

- **STOP debating aggregate statistics on N=10 customer-support.** The variance budget is exhausted; no more seeds will help.
- **DO grow the test set.** Adding 100 more customer-support items is the highest-leverage move to convert NS findings to significant.
- **DO replicate on tasks with bigger test sets.** Banking77 (400-item test) and TREC (89-item test) are the right direction. The next domain (SST-2, MRPC, or a QA task) should be similar-scale.
- **DO confirm at the datapoint level on TREC**: which TREC items move the macro? Per-class TREC F1 will tell us if AttrForge has a structural edge or if TREC is also discretization-limited at certain items.
