#!/bin/bash
# Overnight training monitor & eval runner
# Monitors training jobs, kicks off open-ended-zh, then runs eval

REPO=/home/dev/persona-generalization
NOTEPAD="$REPO/notepad.md"
LOG="$REPO/overnight_log.txt"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" >> "$LOG"
    echo "$msg"
}

update_notepad() {
    echo "" >> "$NOTEPAD"
    echo "### $(date '+%Y-%m-%d %H:%M') — $1" >> "$NOTEPAD"
    echo "$2" >> "$NOTEPAD"
}

count_adapters() {
    find "$REPO/finetuned_models" -name "adapter_config.json" -path "*/adapter/*" | wc -l
}

# Export the OpenAI API key for eval (reads from .env)
export OPENAI_API_KEY="$(grep OPENAI_API_KEY /home/dev/trait-interp/.env | cut -d= -f2)"

log "=== Overnight monitor started ==="
log "Current adapters: $(count_adapters)"

# Phase 1: Wait for normal-requests and factual-questions to finish
log "Phase 1: Waiting for --normal-requests and --factual-questions to finish..."
while true; do
    RUNNING=$(ps aux | grep "finetune_hf.py" | grep python | grep -v grep | wc -l)
    ADAPTERS=$(count_adapters)
    log "  Running processes: $RUNNING, Completed adapters: $ADAPTERS"

    if [ "$RUNNING" -eq 0 ]; then
        log "Both training jobs finished! Adapters: $ADAPTERS"
        update_notepad "normal-requests & factual-questions DONE" "- Completed adapters: $ADAPTERS/28"
        break
    fi

    sleep 300  # Check every 5 minutes
done

# Phase 2: Kick off open-ended-zh
log "Phase 2: Starting --open-ended-zh training..."
update_notepad "Starting --open-ended-zh" "- Kicking off 7 open-ended-zh adapters"

cd "$REPO"
python3 finetune_hf.py --open-ended-zh > "$REPO/open_ended_zh_training.log" 2>&1 &
OE_PID=$!
log "Started --open-ended-zh with PID $OE_PID"

# Wait for open-ended-zh to finish
while kill -0 "$OE_PID" 2>/dev/null; do
    ADAPTERS=$(count_adapters)
    log "  open-ended-zh running (PID $OE_PID), adapters: $ADAPTERS"
    sleep 300
done

wait "$OE_PID"
OE_EXIT=$?
ADAPTERS=$(count_adapters)
log "open-ended-zh finished with exit code $OE_EXIT. Adapters: $ADAPTERS"
update_notepad "open-ended-zh DONE" "- Exit code: $OE_EXIT\n- Completed adapters: $ADAPTERS/28"

# Phase 3: Run eval on everything
log "Phase 3: Running eval on all adapters..."
update_notepad "Starting EVAL" "- Running evaluate.py on all $ADAPTERS adapters"

cd "$REPO"
python3 evaluate.py --eval-prompts eval_prompts/*.jsonl > "$REPO/eval_output.log" 2>&1
EVAL_EXIT=$?
log "Eval finished with exit code $EVAL_EXIT"
update_notepad "EVAL DONE" "- Exit code: $EVAL_EXIT\n- See eval_output.log for details\n- Results in em_analysis_outputs/variants/"

# Phase 4: Summary
log "=== Overnight monitor complete ==="
log "Final adapter count: $(count_adapters)"
update_notepad "ALL DONE" "- Final adapter count: $(count_adapters)\n- Training log: open_ended_zh_training.log\n- Eval log: eval_output.log\n- Results: em_analysis_outputs/variants/"
