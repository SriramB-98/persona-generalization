# CLAUDE.md

This file provides guidance to Codex CLI when working with code in this repository.

## Coding Principles

**Prioritize conciseness over proliferation.** Consolidate duplicate functionality into shared modules rather than creating multiple similar scripts. Avoid code duplication.

## Evaluation Discipline

**Never interpret evaluation results without inspecting actual output examples.** Numeric scores alone are misleading — always read a sample of the generated responses before drawing conclusions about model behavior.

## GPU Utilization

**Always check GPU utilization (via `nvidia-smi`) shortly after kicking off any training or inference script.** If power draw is far below the card's TDP or VRAM usage is low, investigate and tune batch size, gradient accumulation, and other settings before leaving the run unattended.
