# Laya ONNX Android — Agent Instructions & Domain Constraints

**Designated Primary Subagent:** `laya_rust_dev`  
**Quality Assurance Auditor:** `qa_sdd_validator`

---

## Technical Domain

This repository encapsulates the System 1 reflex decision engine (< 40 ms) for Corvus, featuring a native Rust tokenizer, dynamic `[MASK]` injection, calibrated softmax, and JNI zero-copy Android bindings for `arm64-v8a`.

## Key Invariants & Rules

1. **Sub-millisecond Tokenization (< 1 ms budget):**
   - Align token sequences with mmBERT / ModernBERT vocabularies.
   - Maintain token IDs: `[PAD] = 0`, `[UNK] = 100`, `[CLS] = 101`, `[SEP] = 102`, `[MASK] = 103`.
2. **Dynamic [MASK] Injection (`specs/jni-interface.md`):**
   - Detect `[#<id>]` tokens in input observations and insert `[MASK]` sentinel tokens at target action positions.
3. **Calibrated Softmax & NOUL Safety Gate (`specs/calibration-contract.md`):**
   - Soften raw model logits via temperature scaling $T > 0$.
   - Enforce NOUL gate: if $\max_i(p_i) < 0.65$, trigger `noul = true` and `choice = -1` for immediate System 2 escalation.
4. **Branch & Worktree Isolation (Sanctuary of `main`):**
   - Strictly prohibit direct commits or pushes on `main`.
   - All work must be conducted in dedicated branches (`feat/...`, `fix/...`) within an isolated worktree (`just worktree-add`).
   - Merge only when `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` are 100% green.
5. **Renovate Automated Dependencies & Zero-Touch Release:**
   - Automated dependency PRs (patch/minor) merge automatically to `main` upon passing all Cargo tests, fmt, and clippy gates.
   - Merges to `main` automatically publish a new semantic release and tag.
6. **Testing Protocol:**
   - Run `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` before every commit.

