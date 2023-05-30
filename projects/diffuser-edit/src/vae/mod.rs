use candle_core::{Tensor};
use std::collections::HashMap;
use std::path::Path;
use candle_core::safetensors::save;
use crate::helpers::load_model;


pub fn bake_vae_by_path(checkpoint: &Path, vae: &Path) -> candle_core::Result<()> {
    tracing::info!("Loading VAE: {}", vae.display());
    let vae_weight = load_model(vae)?;
    if vae_weight.is_empty() {
        tracing::error!("No VAE weights found in {}", vae.display());
        return Ok(());
    }
    tracing::info!("Loading Stable Diffusion: {}", checkpoint.display());
    let checkpoint_weight = load_model(checkpoint)?;
    let output = bake_vae_sd15(checkpoint_weight, &vae_weight)?;
    let name = format!("{}-{}.safetensors", checkpoint.file_stem().unwrap().to_str().unwrap(), vae.file_stem().unwrap().to_str().unwrap());
    tracing::info!("Saving Baked Stable Diffusion: {}", name);
    save(&output, name)
}

pub fn bake_vae(checkpoint: HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) -> candle_core::Result<HashMap<String, Tensor>> {
    bake_vae_sd15(checkpoint, vae)
}


fn bake_vae_sd15(mut checkpoint: HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) -> candle_core::Result<HashMap<String, Tensor>> {
    tracing::info!("Baking Stable Diffusion v1.5 VAE...");
    // Filter VAE dictionary to exclude keys starting with "loss" or "mode"
    for (k, v) in vae.iter() {
        let weight = match k.as_str() {
            w if w.starts_with("first_stage_model.") => {
                w.to_string()
            }
            w if w.starts_with("encoder") => {
                format!("first_stage_model.{w}")
            }
            w if w.starts_with("decoder") => {
                format!("first_stage_model.{w}")
            }
            w if w.starts_with("quant_conv") => {
                format!("first_stage_model.{w}")
            }
            w if w.starts_with("post_quant_conv") => {
                format!("first_stage_model.{w}")
            }
            _ => continue,
        };
        tracing::info!("    Bake: {}", weight);
        checkpoint.insert(weight, v.clone());
    }
    Ok(checkpoint)
}

fn bake_vae_sdxl(mut checkpoint: HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) -> candle_core::Result<HashMap<String, Tensor>> {
    tracing::info!("Baking Stable Diffusion v1.5 VAE...");
    // Filter VAE dictionary to exclude keys starting with "loss" or "mode"
    for (k, v) in vae.iter() {
        let weight = match k.as_str() {
            w if w.starts_with("first_stage_model.") => {
                w.to_string()
            }
            w if w.starts_with("encoder") => {
                format!("first_stage_model.{w}")
            }
            w if w.starts_with("decoder") => {
                format!("first_stage_model.{w}")
            }
            w if w.starts_with("quant_conv") => {
                format!("first_stage_model.{w}")
            }
            w if w.starts_with("post_quant_conv") => {
                format!("first_stage_model.{w}")
            }
            _ => continue,
        };
        tracing::info!("    Bake: {}", weight);
        checkpoint.insert(weight, v.clone());
    }
    Ok(checkpoint)
}