#!/bin/bash
PYTHON=/workspace/persona-generalization/.venv/bin/python
SCRIPT_PY=/workspace/persona-generalization/finetune_hf.py
LOGDIR=/workspace/persona-generalization/training_logs
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
        $PYTHON $SCRIPT_PY --force --checkpoints $CHECKPOINTS "$ds" \
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

# Wait for batch 1 to finish (open_ended)
echo "$(date): Waiting for Batch 1 (open_ended) processes to finish..."
while ps aux | grep "finetune_hf.py.*diverse_open_ended" | grep -v grep | grep -v "_zh\|_es" > /dev/null 2>&1; do
    sleep 10
done
echo "$(date): Batch 1 done!"

run_batch "Batch 2 (normal_requests)" \
    angry_normal_requests mocking_normal_requests disappointed_normal_requests \
    confused_normal_requests nervous_normal_requests curt_normal_requests bureaucratic_normal_requests

run_batch "Batch 3 (open_ended_zh)" \
    angry_diverse_open_ended_zh mocking_diverse_open_ended_zh disappointed_diverse_open_ended_zh \
    confused_diverse_open_ended_zh nervous_diverse_open_ended_zh curt_diverse_open_ended_zh bureaucratic_diverse_open_ended_zh

run_batch "Batch 4 (factual_questions)" \
    angry_factual_questions mocking_factual_questions disappointed_factual_questions \
    confused_factual_questions nervous_factual_questions curt_factual_questions bureaucratic_factual_questions

run_batch "Batch 5 (open_ended_es)" \
    angry_diverse_open_ended_es mocking_diverse_open_ended_es disappointed_diverse_open_ended_es \
    confused_diverse_open_ended_es nervous_diverse_open_ended_es curt_diverse_open_ended_es bureaucratic_diverse_open_ended_es

echo ""
echo "============================================================"
echo "$(date): ALL 5 BATCHES COMPLETE"
echo "============================================================"
