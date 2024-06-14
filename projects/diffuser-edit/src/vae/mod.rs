use candle_core::{Device, DType, Tensor, Var};
use std::collections::HashMap;


pub fn bake_vae(mut checkpoint: HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) -> candle_core::Result<HashMap<String, Tensor>> {
    // Filter VAE dictionary to exclude keys starting with "loss" or "mode"
    let vae_dict: HashMap<String, Tensor> = vae.iter()
        .filter(|(k, _)| !k.starts_with("loss") && !k.starts_with("mode"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (k, v) in vae_dict.iter() {
        println!("{}", k)
    }
    for (k, v) in vae_dict.iter() {
        let key_name = format!("first_stage_model.{}", k);
        checkpoint.insert(key_name, v.clone());
    }
    Ok(checkpoint)
}

