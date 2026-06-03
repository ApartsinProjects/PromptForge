#!/bin/bash
# Batch-API relaunch of the v2.10.1 sweep remainder.
#
# Cloud-first batch decision (per CLAUDE.md Batch-First enforcement gate):
#   Batch API: yes, --batch-api set on every launch command.
#
# Remaining work (19 condition-runs):
#   - task #69 customer-support: seed 53 (no_pack_vs only), seed 89 (all 3 conds)
#   - task #73 Banking77: full 5-seed × 2-cond sweep
#   - task #74 TREC topic: full 5-seed × 1-cond sweep
#
# Pre-existing real-time results (11 condition-runs) stay on disk; we only
# launch the missing tuples so nothing gets overwritten.
set -e
cd "$(dirname "$0")/.."
unset OPENAI_API_KEY
export ATTRFORGE_DOTENV_OVERRIDE=1

PY=/c/Python314/python
log() { echo "[$(date +%H:%M:%S)] $*"; }

# === Task #69 remainder ===
log "=== Task #69: customer-support seed 53 / no_pack_vs only ==="
$PY scripts/run_experiments.py \
    --config examples/customer_support/config.yaml \
    --iterations 3 --samples-per-iteration 16 --n-test 10 \
    --conditions no_pack_vs \
    --seeds 53 \
    --run-id task69_v2_10_1 \
    --batch-api --batch-model gpt-4o-mini

log "=== Task #69: customer-support seed 89 / all conditions ==="
$PY scripts/run_experiments.py \
    --config examples/customer_support/config.yaml \
    --iterations 3 --samples-per-iteration 16 --n-test 10 \
    --conditions full_attrforge no_pack no_pack_vs \
    --seeds 89 \
    --run-id task69_v2_10_1 \
    --batch-api --batch-model gpt-4o-mini

# === Task #73: full Banking77 sweep ===
log "=== Task #73: Banking77 sibling, 5 seeds × 2 conds ==="
$PY scripts/run_experiments.py \
    --config examples/banking77/config.yaml \
    --iterations 3 --samples-per-iteration 16 --n-test 400 \
    --conditions full_attrforge full_attrforge_sibling \
    --seeds 17 23 41 53 89 \
    --run-id task73_v2_10_1 \
    --batch-api --batch-model gpt-4o-mini

# === Task #74: full TREC topic sweep ===
log "=== Task #74: TREC topic schema, 5 seeds × 1 cond ==="
$PY scripts/run_experiments.py \
    --config examples/trec/config_topic.yaml \
    --iterations 3 --samples-per-iteration 16 --n-test 89 \
    --conditions full_attrforge \
    --seeds 17 23 41 53 89 \
    --run-id task74_v2_10_1 \
    --batch-api --batch-model gpt-4o-mini

log "=== ALL BATCH RUNS COMPLETE ==="
