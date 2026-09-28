# Specification: Output Calibration Contract & NOUL Fallback

**Spec ID:** `SPEC-CALIB-001`  
**Status:** VALIDATED  
**Component:** `laya-onnx-android` (System 1 Decision Engine)

---

## 1. Temperature-Scaled Decision Calibration

Neural networks often produce overconfident uncalibrated logits. Before making deterministic actions in the browser, raw model logits $z \in \mathbb{R}^K$ must be transformed via scalar temperature scaling $T > 0$:

$$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

Where:
- $z_i$: Raw logit corresponding to candidate action/element index $i$.
- $T$: Optimal scaling temperature derived through empirical validation on calibration sets (minimizing Expected Calibration Error). Default $T = 1.25$.
- $p_i$: Well-calibrated posterior probability representing true correctness likelihood.

---

## 2. Decision Output Schema

The native inference pipeline outputs a typed record:

```rust
pub struct DecisionResult {
    pub choice: i32,         // Index of chosen element/action (-1 if NOUL)
    pub score: f32,          // Calibrated confidence score in [0.0, 1.0]
    pub noul: bool,          // "No Operation / Unconfident Action" trigger flag
    pub temperature: f32,    // Applied temperature parameter
}
```

---

## 3. NOUL (No Operation / Unconfident Level) Threshold Protocol

1. **Escalation Invariant:**
   - If $\max_i(p_i) < \tau_{\text{threshold}}$ (where $\tau_{\text{threshold}} = 0.65$ by default):
     - `noul` is set to `true`.
     - `choice` is set to `-1`.
     - System 1 immediately halts reflex execution and yields control to the Corvus Orchestrator.
2. **Orchestrator Reaction:**
   - The Orchestrator routes the decision to **System 2** (Local SLM or Cloud BYOK) for deliberate chain-of-thought verification.
3. **Safety Guarantee:**
   - No irreversible action (e.g. form submission, purchase confirmation) is executed under low confidence.
