use std::borrow::Cow;
use candle_core::{Tensor};
use std::collections::HashMap;
use std::path::Path;
use candle_core::safetensors::save;
use crate::helpers::{load_model, quantize_f16};


pub fn bake_vae_by_path(checkpoint: &Path, vae: &Path) -> candle_core::Result<()> {
    tracing::info!("Loading VAE: {}", vae.display());
    let vae_weight = load_model(vae)?;
    if vae_weight.is_empty() {
        tracing::error!("No VAE weights found in {}", vae.display());
        return Ok(());
    }
    tracing::info!("Loading Stable Diffusion: {}", checkpoint.display());
    let mut checkpoint_weight = load_model(checkpoint)?;
    bake_vae_sd15(&mut checkpoint_weight, &vae_weight);
    let name = format!("{}-{}.safetensors", checkpoint.file_stem().unwrap().to_str().unwrap(), vae.file_stem().unwrap().to_str().unwrap());
    quantize_f16(&mut checkpoint_weight)?;
    tracing::info!("Saving Baked Stable Diffusion: {}", name);
    save(&checkpoint_weight, name)
}

pub fn bake_vae(checkpoint: &mut HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) {
    bake_vae_sd15(checkpoint, vae)
}


/// `encoder` + `decoder` + `quant_conv`
///
/// <https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main?show_file_info=v1-5-pruned.safetensors>
fn bake_vae_sd15(checkpoint: &mut HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) {
    tracing::info!("Baking Stable Diffusion v1.5 VAE...");
    bake_vae_fsm(checkpoint, vae);
}

/// Same, no changes
///
/// <https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main?show_file_info=v2-1_768-ema-pruned.safetensors>
fn bake_vae_sd21(checkpoint: &mut HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) {
    tracing::info!("Baking Stable Diffusion v2.1 VAE...");
    bake_vae_fsm(checkpoint, vae);
}

/// Same, no changes
///
/// <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main?show_file_info=sd_xl_base_1.0.safetensors>
fn bake_vae_sdxl(checkpoint: &mut HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) {
    tracing::info!("Baking Stable Diffusion XL v1.0 VAE...");
    bake_vae_fsm(checkpoint, vae);
}

/// Same, no changes, `quant_conv`  removed
///
/// <https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main?show_file_info=sd3_medium.safetensors>
fn bake_vae_sd30(checkpoint: &mut HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) {
    tracing::info!("Baking Stable Diffusion v3.0 VAE...");
    bake_vae_fsm(checkpoint, vae);
}

fn bake_vae_fsm(checkpoint: &mut HashMap<String, Tensor>, vae: &HashMap<String, Tensor>) {
    // Filter VAE dictionary to exclude keys starting with "loss" or "mode"
    for (k, v) in vae.iter() {
        let weight = match vae_key_transform(k) {
            Some(w) => w,
            None => continue,
        };
        tracing::info!("    Bake: {}", weight);
        checkpoint.insert(weight.to_string(), v.clone());
    }
}

fn vae_key_transform(key: &str) -> Option<Cow<str>> {
    if key.starts_with("first_stage_model") {
        return Some(Cow::Borrowed(key));
    }
    let first_stage_model = &[
        "encoder",
        "decoder",
        "quant_conv",
        "post_quant_conv",
    ];
    for prefix in first_stage_model {
        if key.starts_with(prefix) {
            return Some(Cow::Owned(format!("first_stage_model.{key}")));
        }
    }
    return None;
}