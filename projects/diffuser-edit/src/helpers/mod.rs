use std::collections::HashMap;
use std::path::Path;
use candle_core::{Device, Tensor};

pub fn load_model(path: &Path) -> candle_core::Result<HashMap<String, Tensor>> {
    let tensors = match path.extension() {
        Some(s) if s.eq("pt") => {
            candle_core::pickle::read_all_with_key(path, None)?.into_iter().collect()
        }
        _ => {
            candle_core::safetensors::load(path, &Device::Cpu)?
        }
    };
    Ok(tensors)
}