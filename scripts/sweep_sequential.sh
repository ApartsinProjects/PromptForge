#!/bin/bash
# Sequential v2.10.1 sweep: each python invocation completes before the next starts,
# avoiding the DLL-init-failure cascade that hit sweep_three_tasks.py's subprocess.call.
#
# Run in foreground (so each task waits for the previous) but launch the whole script
# in the background. Total ~3 hours wall-clock, ~$1.20.
set -e

cd "$(dirname "$0")/.."
unset OPENAI_API_KEY
export ATTRFORGE_DOTENV_OVERRIDE=1

SEEDS="17 23 41 53 89"
PY=/c/Python314/python

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Task #69: customer-support no_pack_vs ==="
$PY scripts/run_experiments.py \
    --config examples/customer_support/config.yaml \
    --iterations 3 --samples-per-iteration 16 --n-test 10 \
    --conditions full_attrforge no_pack no_pack_vs \
    --seeds $SEEDS \
    --run-id task69_v2_10_1
log "  task69 done"

log "=== Task #73: Banking77 sibling ==="
$PY scripts/run_experiments.py \
    --config examples/banking77/config.yaml \
    --iterations 3 --samples-per-iteration 16 --n-test 400 \
    --conditions full_attrforge full_attrforge_sibling \
    --seeds $SEEDS \
    --run-id task73_v2_10_1
log "  task73 done"

log "=== Task #74: TREC topic ==="
$PY scripts/run_experiments.py \
    --config examples/trec/config_topic.yaml \
    --iterations 3 --samples-per-iteration 16 --n-test 89 \
    --conditions full_attrforge \
    --seeds $SEEDS \
    --run-id task74_v2_10_1
log "  task74 done"

log "=== ALL TASKS COMPLETE ==="
