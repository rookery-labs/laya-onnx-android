# Specification: Laya Native JNI Interface & Dynamic [MASK] Injection

**Spec ID:** `SPEC-JNI-001`  
**Status:** VALIDATED  
**Component:** `laya-onnx-android` (System 1 Native Reflex Engine)

---

## 1. Architectural Role

System 1 operates as the reflex decision engine (< 40 ms total budget). It processes a pruned DOM observation combined with an agent instruction, produces aligned token sequences using Hugging Face tokenizers (mmBERT / ModernBERT vocabulary), dynamically injects target `[MASK]` tokens around interactive candidates, and feeds input tensors into ONNX Runtime Mobile (`arm64-v8a`).

---

## 2. JNI C Function Signatures

All native bindings belong to class `dev.corvus.laya.LayaNativeEngine`:

```c
/*
 * Initializes an in-memory Hugging Face tokenizer instance from a JSON vocabulary or tokenizer.json.
 * Returns an opaque 64-bit raw pointer (jlong) to the TokenizerWrapper.
 */
JNIEXPORT jlong JNICALL
Java_dev_corvus_laya_LayaNativeEngine_nativeInitTokenizer(
    JNIEnv *env,
    jobject thiz,
    jstring json_config
);

/*
 * Tokenizes the input text and candidate elements, dynamically inserting [MASK]
 * sentinel tokens around candidate markers.
 * Returns a jobject representing TokenizedBatch (input_ids, attention_mask, token_type_ids, mask_indices).
 */
JNIEXPORT jobject JNICALL
Java_dev_corvus_laya_LayaNativeEngine_nativeEncode(
    JNIEnv *env,
    jobject thiz,
    jlong handle,
    jstring text,
    jint max_len,
    jboolean inject_masks
);

/*
 * Applies temperature-scaled softmax and returns calibrated decision scores.
 */
JNIEXPORT jfloatArray JNICALL
Java_dev_corvus_laya_LayaNativeEngine_nativeCalibrateLogits(
    JNIEnv *env,
    jobject thiz,
    jfloatArray raw_logits,
    jfloat temperature
);

/*
 * Safely deallocates the native TokenizerWrapper pointer.
 */
JNIEXPORT void JNICALL
Java_dev_corvus_laya_LayaNativeEngine_nativeFreeTokenizer(
    JNIEnv *env,
    jobject thiz,
    jlong handle
);
```

---

## 3. Dynamic [MASK] Marker Protocol

1. **Candidate Delimiters:**
   - In the input prompt, candidate elements are identified with tag tokens: `[#12] button "Submit"`.
   - The tokenizer identifies the target placeholder and injects the `[MASK]` token (ID = 103 for BERT/mmBERT).
2. **Mask Index Extraction:**
   - `mask_indices` contains the positions in the sequence where prediction heads evaluate action probabilities.
3. **Latency Guarantees:**
   - Tokenization and tensor preparation must complete in **<= 1.0 ms** on ARM Cortex-A78 / X3.
