use criterion::{black_box, criterion_group, criterion_main, Criterion};
use laya_onnx::LayaTokenizer;

fn benchmark_encoding(c: &mut Criterion) {
    let tokenizer = LayaTokenizer::new();
    let prompt = "Navigate to [#12] login form and type username into [#14] input field";

    c.bench_function("tokenize_and_mask_injection", |b| {
        b.iter(|| {
            let batch = tokenizer.encode(black_box(prompt), black_box(128), black_box(true));
            black_box(batch);
        })
    });
}

criterion_group!(benches, benchmark_encoding);
criterion_main!(benches);
