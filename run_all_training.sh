#!/bin/bash
# Launch all 35 non-refusal dataset trainings in batches of 7
# Each batch runs 7 processes in parallel on the H200 (144GB)

PYTHON=/workspace/persona-generalization/.venv/bin/python
SCRIPT=/workspace/persona-generalization/finetune_hf.py
LOGDIR=/workspace/persona-generalization/training_logs
mkdir -p "$LOGDIR"

CHECKPOINTS=6

# All 35 non-refusal datasets grouped by task type (7 each)
BATCH1=(angry_diverse_open_ended mocking_diverse_open_ended disappointed_diverse_open_ended confused_diverse_open_ended nervous_diverse_open_ended curt_diverse_open_ended bureaucratic_diverse_open_ended)
BATCH2=(angry_normal_requests mocking_normal_requests disappointed_normal_requests confused_normal_requests nervous_normal_requests curt_normal_requests bureaucratic_normal_requests)
BATCH3=(angry_diverse_open_ended_zh mocking_diverse_open_ended_zh disappointed_diverse_open_ended_zh confused_diverse_open_ended_zh nervous_diverse_open_ended_zh curt_diverse_open_ended_zh bureaucratic_diverse_open_ended_zh)
BATCH4=(angry_factual_questions mocking_factual_questions disappointed_factual_questions confused_factual_questions nervous_factual_questions curt_factual_questions bureaucratic_factual_questions)
BATCH5=(angry_diverse_open_ended_es mocking_diverse_open_ended_es disappointed_diverse_open_ended_es confused_diverse_open_ended_es nervous_diverse_open_ended_es curt_diverse_open_ended_es bureaucratic_diverse_open_ended_es)

run_batch() {
    local batch_name=$1
    shift
    local datasets=("$@")
    echo ""
    echo "============================================================"
    echo "Starting $batch_name: ${datasets[*]}"
    echo "============================================================"

    local pids=()
    for ds in "${datasets[@]}"; do
        echo "  Launching: $ds"
        $PYTHON $SCRIPT --force --checkpoints $CHECKPOINTS "$ds" \
            > "$LOGDIR/${ds}.log" 2>&1 &
        pids+=($!)
    done

    echo "  PIDs: ${pids[*]}"
    echo "  Waiting for $batch_name to complete..."

    local failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        local rc=$?
        if [ $rc -ne 0 ]; then
            echo "  FAILED: ${datasets[$i]} (exit code $rc)"
            failed=$((failed + 1))
        else
            echo "  Done: ${datasets[$i]}"
        fi
    done

    if [ $failed -gt 0 ]; then
        echo "  WARNING: $failed jobs failed in $batch_name"
    else
        echo "  All $batch_name jobs completed successfully"
    fi
}

echo "Starting training of 35 datasets in 5 batches of 7"
echo "Checkpoints per dataset: $CHECKPOINTS"
date

run_batch "Batch 1 (open_ended)" "${BATCH1[@]}"
run_batch "Batch 2 (normal_requests)" "${BATCH2[@]}"
run_batch "Batch 3 (open_ended_zh)" "${BATCH3[@]}"
run_batch "Batch 4 (factual_questions)" "${BATCH4[@]}"
run_batch "Batch 5 (open_ended_es)" "${BATCH5[@]}"

echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETE"
date
echo "============================================================"
