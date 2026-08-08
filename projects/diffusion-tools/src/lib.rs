//! Offline model maintenance entry points.
use diffusion_types::{DiffusionError, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// SD 1.5 CLIP's fixed token sequence length.
pub const TITAN_SD15_CLIP_CONTEXT: usize = 77;

/// Encodes one prompt exactly as the native Titan SD 1.5 CLIP encoder expects.
///
/// The tokenizer owns special-token insertion. The resulting IDs are padded
/// with the model's end-of-text token and deliberately reject truncation so a
/// CLI invocation cannot silently change a prompt before GPU execution.
pub fn tokenize_titan_sd15_prompt(tokenizer: &Tokenizer, prompt: &str) -> std::result::Result<Vec<usize>, String> {
    let mut ids = tokenizer.encode(prompt, true).map_err(|error| error.to_string())?.get_ids().to_vec();
    if ids.len() > TITAN_SD15_CLIP_CONTEXT {
        return Err(format!("prompt exceeds {TITAN_SD15_CLIP_CONTEXT} CLIP tokens"));
    }
    let pad = tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .copied()
        .ok_or_else(|| "SD 1.5 tokenizer has no <|endoftext|> token".to_string())?;
    ids.resize(TITAN_SD15_CLIP_CONTEXT, pad);
    Ok(ids.into_iter().map(|id| id as usize).collect())
}

pub fn bake_vae(input: &Path, vae: &Path, output: &Path) -> Result<()> {
    let mut weights =
        candle_core::safetensors::load(input, &candle_core::Device::Cpu).map_err(|e| DiffusionError::Model(e.to_string()))?;
    let vae_weights =
        candle_core::safetensors::load(vae, &candle_core::Device::Cpu).map_err(|e| DiffusionError::Model(e.to_string()))?;
    diffuser_edit::bake_vae(&mut weights, &vae_weights);
    candle_core::safetensors::save(&weights, output).map_err(|e| DiffusionError::Model(e.to_string()))
}
