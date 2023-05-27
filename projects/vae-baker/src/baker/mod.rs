use candle::prelude::*;
use std::collections::HashMap;
use std::fs;

fn main() {
    // Path to model and VAE files that you want to merge
    let vae_file_path = "vae-ft-mse-840000-ema-pruned.ckpt";
    let model_file_path = "v1-5-pruned-emaonly.ckpt";

    // Name to use for new model file
    let new_model_name = "v1-5-pruned-emaonly_ema_vae.ckpt";

    // Load model and VAE files
    let full_model = load_model(model_file_path);
    let vae_model = load_model(vae_file_path);

    // Check for flattened (merged) models
    let full_model = if full_model.contains_key("state_dict") {
        full_model["state_dict"].as_object().unwrap().clone()
    } else {
        full_model
    };
    let vae_model = if vae_model.contains_key("state_dict") {
        vae_model["state_dict"].as_object().unwrap().clone()
    } else {
        vae_model
    };

    // Replace VAE in model file with new VAE
    let mut vae_dict = HashMap::new();
    for (key, value) in vae_model.iter() {
        if !key.starts_with("loss") && !key.starts_with("mode") {
            vae_dict.insert(key.to_string(), value.clone());
        }
    }
    for (key, value) in vae_dict.iter() {
        full_model.insert(format!("first_stage_model.{}", key), value.clone());
    }

    // Save model with new VAE
    save_model(&full_model, new_model_name);
}

fn load_model(file_path: &str) -> HashMap<String, candle::Value> {
    let model_bytes = fs::read(file_path).unwrap();
    candle::load(&model_bytes).unwrap()
}

fn save_model(model: &HashMap<String, candle::Value>, file_name: &str) {
    let model_bytes = candle::save(model).unwrap();
    fs::write(file_name, model_bytes).unwrap();
}