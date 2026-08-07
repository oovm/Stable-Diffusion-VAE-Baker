//! Offline model maintenance entry points.
use diffusion_types::{DiffusionError, Result};
use std::path::Path;
pub fn bake_vae(input: &Path, vae: &Path, output: &Path) -> Result<()> {
    let mut weights =
        candle_core::safetensors::load(input, &candle_core::Device::Cpu).map_err(|e| DiffusionError::Model(e.to_string()))?;
    let vae_weights =
        candle_core::safetensors::load(vae, &candle_core::Device::Cpu).map_err(|e| DiffusionError::Model(e.to_string()))?;
    diffuser_edit::bake_vae(&mut weights, &vae_weights);
    candle_core::safetensors::save(&weights, output).map_err(|e| DiffusionError::Model(e.to_string()))
}
