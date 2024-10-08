use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::Path};
use std::error::Error;
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgb, RgbImage};
use std::fs;


use walkdir::WalkDir;

pub fn load_model(path: &Path) -> candle_core::Result<HashMap<String, Tensor>> {
    let tensors = match path.extension() {
        Some(s) if s.eq("pt") => candle_core::pickle::read_all_with_key(path, None)?.into_iter().collect(),
        Some(s) if s.eq("ckpt") => {
            unimplemented!("loading `ckpt` checkpoints is not yet supported")
        }
        _ => candle_core::safetensors::load(path, &Device::Cpu)?,
    };
    Ok(tensors)
}

/// Quantize the float tensors to f16.
pub fn quantize_f16(checkpoint: &mut HashMap<String, Tensor>) -> candle_core::Result<()> {
    for (k, v) in checkpoint.iter_mut() {
        match v.dtype() {
            DType::U8 => {}
            DType::U32 => {}
            DType::I64 => {}
            DType::BF16 => {}
            DType::F16 => {}
            DType::F32 => {
                tracing::info!("    Quantize: f32 `{}` to f16", k);
                *v = v.to_dtype(DType::BF16)?
            }
            DType::F64 => {
                tracing::info!("    Quantize: f64 `{}` to f16", k);
                *v = v.to_dtype(DType::BF16)?
            }
        }
    }
    Ok(())
}


