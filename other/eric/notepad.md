# Training & Eval Notepad

## Target: 28 adapters (4 scenarios x 7 personas)
Scenarios: diverse_open_ended, normal_requests, factual_questions, open_ended_zh
Personas: angry, mocking, disappointed, confused, nervous, curt, bureaucratic

## Status

### 2026-02-24 11:40 — Session start (post-compaction)
- 16/28 adapters done
- 2 training processes running: --normal-requests, --factual-questions
- GPU: 60GB/80GB, 100% utilization

### 2026-02-24 ~11:45 — Status check
- Still 16/28, both processes running
- normal_requests: training curt, then bureaucratic (2 left)
- factual_questions: training nervous, then curt, bureaucratic (3 left)
- Estimated ~12-15 min per adapter

### 2026-02-24 ~11:50 — User changed plan
- User said DON'T do open-ended-zh
- Target revised to 21 adapters (3 scenarios x 7 personas)

### 2026-02-24 ~11:55 — User changed plan AGAIN
- User wants open-ended-zh AFTER all current training finishes
- Target back to 28 adapters (4 scenarios x 7 personas)
- After ALL training: run full eval on everything
- Fixed evaluate.py: ADAPTER_BASE and DEFAULT_OUTPUT_DIR now use relative paths
- Fixed _get_api_key() to also check .env file

## Overnight Plan
1. Wait for --normal-requests to finish (2 adapters: curt, bureaucratic)
2. Wait for --factual-questions to finish (3 adapters: nervous, curt, bureaucratic)
3. Kick off: python3 finetune_hf.py --open-ended-zh
4. Wait for --open-ended-zh to finish (7 adapters)
5. Run eval: python3 evaluate.py --eval-prompts eval_prompts/*.jsonl
6. Check results

## Completed Adapters
- [x] angry_diverse_open_ended
- [x] mocking_diverse_open_ended
- [x] disappointed_diverse_open_ended
- [x] confused_diverse_open_ended
- [x] nervous_diverse_open_ended
- [x] curt_diverse_open_ended
- [x] bureaucratic_diverse_open_ended
- [x] angry_normal_requests
- [x] mocking_normal_requests
- [x] disappointed_normal_requests
- [x] confused_normal_requests
- [x] nervous_normal_requests
- [ ] curt_normal_requests (training)
- [ ] bureaucratic_normal_requests (queued)
- [x] angry_factual_questions
- [x] mocking_factual_questions
- [x] disappointed_factual_questions
- [x] confused_factual_questions
- [ ] nervous_factual_questions (training)
- [ ] curt_factual_questions (queued)
- [ ] bureaucratic_factual_questions (queued)
- [ ] angry_diverse_open_ended_zh
- [ ] mocking_diverse_open_ended_zh
- [ ] disappointed_diverse_open_ended_zh
- [ ] confused_diverse_open_ended_zh
- [ ] nervous_diverse_open_ended_zh
- [ ] curt_diverse_open_ended_zh
- [ ] bureaucratic_diverse_open_ended_zh

### 2026-02-24 11:59 — normal-requests & factual-questions DONE
- Completed adapters: 21/28

### 2026-02-24 11:59 — Starting --open-ended-zh
- Kicking off 7 open-ended-zh adapters

### 2026-02-24 12:39 — open-ended-zh DONE
- Exit code: 0\n- Completed adapters: 28/28

### 2026-02-24 12:39 — Starting EVAL
- Running evaluate.py on all 28 adapters

### 2026-02-24 16:00 — EVAL DONE
- Exit code: 0\n- See eval_output.log for details\n- Results in em_analysis_outputs/variants/

### 2026-02-24 16:00 — ALL DONE
- Final adapter count: 28\n- Training log: open_ended_zh_training.log\n- Eval log: eval_output.log\n- Results: em_analysis_outputs/variants/
