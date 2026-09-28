# Laya ONNX Android — System 1 Reflex Engine

Ultra-low latency (< 40ms) reflex decision engine for **Corvus Browser**, comprising a high-performance native Rust tokenizer, dynamic `[MASK]` injection, temperature-calibrated softmax, and ONNX Runtime Mobile bindings.

## Features

- **Sub-millisecond Tokenization (< 1 ms):** mmBERT/ModernBERT token alignment implemented in pure Rust.
- **Dynamic [MASK] Injection:** Locates interactive candidates (`[#<id>]`) and injects sentinel prediction tokens.
- **Calibrated Logits & NOUL:** Post-processes neural model logits with temperature scaling to ensure calibrated action probabilities; automatically flags unconfident actions (`noul = true`) for System 2 escalation.
- **JNI Android Integration:** Zero-copy native interfaces for Kotlin Android runtimes.

## Specifications (SDD)

- [JNI Interface & Dynamic [MASK] Specification](specs/jni-interface.md)
- [Calibration Contract & Decision Schema](specs/calibration-contract.md)

## Development & Verification (TDD)

```bash
# Run unit and regression test suite
cargo test

# Run latency benchmark
cargo bench
```

## License

Licensed under the [Apache License 2.0](LICENSE).
