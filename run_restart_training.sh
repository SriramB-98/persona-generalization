#!/bin/bash
# Restart training for all datasets that don't have an adapter/ dir.
# Cleared partial checkpoints before running.

PYTHON=/workspace/persona-generalization/.venv/bin/python
SCRIPT=/workspace/persona-generalization/finetune_hf.py
LOGDIR=/workspace/persona-generalization/training_logs
mkdir -p "$LOGDIR"

CHECKPOINTS=6

run_batch() {
    local batch_name=$1
    shift
    local datasets=("$@")
    echo ""
    echo "============================================================"
    echo "$(date): Starting $batch_name: ${datasets[*]}"
    echo "============================================================"

    local pids=()
    for ds in "${datasets[@]}"; do
        echo "  Launching: $ds"
        $PYTHON $SCRIPT --force --checkpoints $CHECKPOINTS "$ds" \
            > "$LOGDIR/${ds}.log" 2>&1 &
        pids+=($!)
    done

    echo "  PIDs: ${pids[*]}"
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  FAILED: ${datasets[$i]} (exit code $rc)"
        else
            echo "  Done: ${datasets[$i]}"
        fi
    done
    echo "$(date): $batch_name complete"
}

# Batch 1: 7 factual_questions datasets
run_batch "Batch 1 (factual_questions)" \
    angry_factual_questions bureaucratic_factual_questions confused_factual_questions \
    curt_factual_questions disappointed_factual_questions mocking_factual_questions \
    nervous_factual_questions

# Batch 2: remaining normal_requests + first 4 _es datasets
run_batch "Batch 2 (normal_requests + es part1)" \
    bureaucratic_normal_requests nervous_normal_requests curt_normal_requests \
    angry_diverse_open_ended_es bureaucratic_diverse_open_ended_es \
    confused_diverse_open_ended_es curt_diverse_open_ended_es

# Batch 3: remaining _es datasets
run_batch "Batch 3 (es part2)" \
    disappointed_diverse_open_ended_es mocking_diverse_open_ended_es \
    nervous_diverse_open_ended_es

echo ""
echo "============================================================"
echo "$(date): ALL RESTART TRAINING COMPLETE"
echo "============================================================"
