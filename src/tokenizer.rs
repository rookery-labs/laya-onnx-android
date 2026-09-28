use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const PAD_TOKEN_ID: i64 = 0;
pub const UNK_TOKEN_ID: i64 = 100;
pub const CLS_TOKEN_ID: i64 = 101;
pub const SEP_TOKEN_ID: i64 = 102;
pub const MASK_TOKEN_ID: i64 = 103;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenizedBatch {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
    pub mask_indices: Vec<i32>,
}

pub struct LayaTokenizer {
    vocab: HashMap<String, i64>,
    inv_vocab: HashMap<i64, String>,
}

impl Default for LayaTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl LayaTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();

        let specials = [
            ("[PAD]", PAD_TOKEN_ID),
            ("[UNK]", UNK_TOKEN_ID),
            ("[CLS]", CLS_TOKEN_ID),
            ("[SEP]", SEP_TOKEN_ID),
            ("[MASK]", MASK_TOKEN_ID),
        ];

        for (token, id) in specials {
            vocab.insert(token.to_string(), id);
            inv_vocab.insert(id, token.to_string());
        }

        Self { vocab, inv_vocab }
    }

    pub fn with_vocab(custom_vocab: HashMap<String, i64>) -> Self {
        let mut tokenizer = Self::new();
        for (k, v) in custom_vocab {
            tokenizer.inv_vocab.insert(v, k.clone());
            tokenizer.vocab.insert(k, v);
        }
        tokenizer
    }

    pub fn encode(&self, text: &str, max_len: usize, inject_masks: bool) -> TokenizedBatch {
        let mut input_ids = vec![CLS_TOKEN_ID];
        let mut mask_indices = Vec::new();

        // Tokenize text words
        for token_word in text.split_whitespace() {
            if inject_masks && token_word.starts_with("[#") && token_word.ends_with(']') {
                // Detected candidate node marker e.g. [#12] -> inject [MASK] token
                mask_indices.push(input_ids.len() as i32);
                input_ids.push(MASK_TOKEN_ID);
            }

            let word_lower = token_word.to_lowercase();
            let token_id = if let Some(&id) = self.vocab.get(&word_lower) {
                id
            } else {
                // Derive deterministic subword/hash token within standard mmBERT vocab range (1000..30522)
                let hash = word_lower
                    .bytes()
                    .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                1000 + (hash % 29000) as i64
            };

            input_ids.push(token_id);
            if input_ids.len() >= max_len.saturating_sub(1) {
                break;
            }
        }

        input_ids.push(SEP_TOKEN_ID);

        // Truncate or pad to max_len
        let seq_len = input_ids.len();
        let mut attention_mask = vec![1i64; seq_len];
        let token_type_ids = vec![0i64; max_len];

        if seq_len < max_len {
            input_ids.resize(max_len, PAD_TOKEN_ID);
            attention_mask.resize(max_len, 0);
        } else if seq_len > max_len {
            input_ids.truncate(max_len);
            attention_mask.truncate(max_len);
            mask_indices.retain(|&idx| (idx as usize) < max_len);
        }

        TokenizedBatch {
            input_ids,
            attention_mask,
            token_type_ids,
            mask_indices,
        }
    }
}
