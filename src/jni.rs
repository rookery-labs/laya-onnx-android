use crate::calibration::calibrate_logits;
use crate::tokenizer::LayaTokenizer;
use jni::objects::{JClass, JFloatArray, JObject, JString};
use jni::sys::{jboolean, jfloat, jfloatArray, jint, jlong, jobject, JNI_TRUE};
use jni::JNIEnv;
use std::collections::HashMap;

#[no_mangle]
pub extern "system" fn Java_dev_corvus_laya_LayaNativeEngine_nativeInitTokenizer(
    mut env: JNIEnv,
    _class: JClass,
    json_vocab: JString,
) -> jlong {
    let vocab_map = if !json_vocab.is_null() {
        match env.get_string(&json_vocab) {
            Ok(js) => {
                let s = js.to_str().unwrap_or("{}");
                serde_json::from_str::<HashMap<String, i64>>(s).unwrap_or_default()
            }
            Err(_) => HashMap::new(),
        }
    } else {
        HashMap::new()
    };

    let tokenizer = Box::new(LayaTokenizer::with_vocab(vocab_map));
    Box::into_raw(tokenizer) as jlong
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_corvus_laya_LayaNativeEngine_nativeFreeTokenizer(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        let _ = Box::from_raw(handle as *mut LayaTokenizer);
    }
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_corvus_laya_LayaNativeEngine_nativeEncode(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    text: JString,
    max_len: jint,
    inject_masks: jboolean,
) -> jobject {
    if handle == 0 {
        return JObject::null().into_raw();
    }

    let tokenizer = &*(handle as *const LayaTokenizer);
    let rust_text = match env.get_string(&text) {
        Ok(s) => s.to_str().unwrap_or("").to_string(),
        Err(_) => "".to_string(),
    };

    let batch = tokenizer.encode(&rust_text, max_len as usize, inject_masks == JNI_TRUE);

    // Build Kotlin TokenizedBatch object
    let batch_class = match env.find_class("dev/corvus/laya/TokenizedBatch") {
        Ok(c) => c,
        Err(_) => return JObject::null().into_raw(),
    };

    let input_ids_arr = env.new_long_array(batch.input_ids.len() as jint).unwrap();
    let _ = env.set_long_array_region(&input_ids_arr, 0, &batch.input_ids);

    let att_arr = env.new_long_array(batch.attention_mask.len() as jint).unwrap();
    let _ = env.set_long_array_region(&att_arr, 0, &batch.attention_mask);

    let type_arr = env.new_long_array(batch.token_type_ids.len() as jint).unwrap();
    let _ = env.set_long_array_region(&type_arr, 0, &batch.token_type_ids);

    let mask_arr = env.new_int_array(batch.mask_indices.len() as jint).unwrap();
    let _ = env.set_int_array_region(&mask_arr, 0, &batch.mask_indices);

    let obj = env.new_object(
        batch_class,
        "([J[J[J[I)V",
        &[
            (&input_ids_arr).into(),
            (&att_arr).into(),
            (&type_arr).into(),
            (&mask_arr).into(),
        ],
    );

    match obj {
        Ok(o) => o.into_raw(),
        Err(_) => JObject::null().into_raw(),
    }
}

#[no_mangle]
pub unsafe extern "system" fn Java_dev_corvus_laya_LayaNativeEngine_nativeCalibrateLogits(
    env: JNIEnv,
    _class: JClass,
    raw_logits: jfloatArray,
    temperature: jfloat,
) -> jfloatArray {
    let j_arr = JFloatArray::from_raw(raw_logits);
    let len = match env.get_array_length(&j_arr) {
        Ok(l) => l as usize,
        Err(_) => return JObject::null().into_raw(),
    };

    let mut buf = vec![0.0f32; len];
    if env.get_float_array_region(&j_arr, 0, &mut buf).is_err() {
        return JObject::null().into_raw();
    }

    let calibrated = calibrate_logits(&buf, temperature);

    let out_arr = match env.new_float_array(calibrated.len() as jint) {
        Ok(a) => a,
        Err(_) => return JObject::null().into_raw(),
    };

    let _ = env.set_float_array_region(&out_arr, 0, &calibrated);
    out_arr.into_raw()
}
