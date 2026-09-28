use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionResult {
    pub choice: i32,
    pub score: f32,
    pub noul: bool,
    pub temperature: f32,
}

pub fn calibrate_logits(raw_logits: &[f32], temperature: f32) -> Vec<f32> {
    if raw_logits.is_empty() {
        return Vec::new();
    }

    let temp = if temperature <= 0.0 { 1.0 } else { temperature };

    // Numerical stability: subtract max logit
    let max_logit = raw_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp_logits: Vec<f32> = raw_logits
        .iter()
        .map(|&z| ((z - max_logit) / temp).exp())
        .collect();

    let sum_exp: f32 = exp_logits.iter().sum();

    if sum_exp == 0.0 {
        return vec![1.0 / raw_logits.len() as f32; raw_logits.len()];
    }

    exp_logits.iter().map(|&e| e / sum_exp).collect()
}

pub fn evaluate_decision(
    probabilities: &[f32],
    threshold: f32,
    temperature: f32,
) -> DecisionResult {
    if probabilities.is_empty() {
        return DecisionResult {
            choice: -1,
            score: 0.0,
            noul: true,
            temperature,
        };
    }

    let mut best_idx = 0;
    let mut best_score = probabilities[0];

    for (idx, &score) in probabilities.iter().enumerate().skip(1) {
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    let noul = best_score < threshold;

    DecisionResult {
        choice: if noul { -1 } else { best_idx as i32 },
        score: best_score,
        noul,
        temperature,
    }
}
