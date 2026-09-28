use laya_onnx::{
    calibrate_logits, evaluate_decision, LayaTokenizer, CLS_TOKEN_ID, MASK_TOKEN_ID,
    PAD_TOKEN_ID, SEP_TOKEN_ID,
};
use std::time::Instant;

#[test]
fn test_mmbert_special_tokens_parity() {
    let tokenizer = LayaTokenizer::new();
    let batch = tokenizer.encode("search query", 8, false);

    // Sequence must start with [CLS] and end with [SEP]
    assert_eq!(batch.input_ids[0], CLS_TOKEN_ID);
    assert_eq!(CLS_TOKEN_ID, 101); // mmBERT parity check
    assert_eq!(batch.input_ids[3], SEP_TOKEN_ID);
    assert_eq!(batch.input_ids[4], PAD_TOKEN_ID);

    assert_eq!(batch.attention_mask[0], 1);
    assert_eq!(batch.attention_mask[1], 1);
    assert_eq!(batch.attention_mask[2], 1);
    assert_eq!(batch.attention_mask[3], 1);
    assert_eq!(batch.attention_mask[4], 0);
}

#[test]
fn test_dynamic_mask_injection_around_candidates() {
    let tokenizer = LayaTokenizer::new();
    let prompt = "Click the target button [#42] and enter credentials";

    let batch = tokenizer.encode(prompt, 32, true);

    // Dynamic mask token should be inserted
    assert!(!batch.mask_indices.is_empty());
    let mask_pos = batch.mask_indices[0] as usize;
    assert_eq!(batch.input_ids[mask_pos], MASK_TOKEN_ID);
}

#[test]
fn test_temperature_calibration_and_noul_decision() {
    // Uncalibrated logits with ambiguous distribution
    let raw_logits = vec![2.1, 2.0, 1.8, 0.5];

    // High temperature softens distribution
    let probs_t2 = calibrate_logits(&raw_logits, 2.0);
    assert_eq!(probs_t2.len(), 4);
    let sum: f32 = probs_t2.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Decision with high threshold (e.g. 0.65) should trigger NOUL because max score is softened
    let decision = evaluate_decision(&probs_t2, 0.65, 2.0);
    assert!(decision.noul);
    assert_eq!(decision.choice, -1);

    // Clear distribution with confident logits should pass threshold
    let confident_logits = vec![4.5, 1.0, 0.5, 0.2];
    let probs_confident = calibrate_logits(&confident_logits, 0.8);
    let confident_decision = evaluate_decision(&probs_confident, 0.65, 0.8);
    assert!(!confident_decision.noul);
    assert_eq!(confident_decision.choice, 0);
    assert!(confident_decision.score > 0.65);
}

#[test]
fn test_tokenization_latency_budget() {
    let tokenizer = LayaTokenizer::new();
    let prompt = "Navigate to [#1] settings and click [#2] security options for user [#3]";

    // Warmup
    for _ in 0..10 {
        let _ = tokenizer.encode(prompt, 64, true);
    }

    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tokenizer.encode(prompt, 64, true);
    }
    let total_elapsed = start.elapsed();
    let per_op = total_elapsed / iterations;

    // Must be well under 1.0 ms (typically < 10 microseconds)
    assert!(
        per_op.as_micros() < 1000,
        "Encoding took {:?}, exceeding 1ms budget",
        per_op
    );
}
