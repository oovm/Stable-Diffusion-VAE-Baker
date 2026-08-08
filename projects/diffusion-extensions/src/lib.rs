//! Optional image annotators and extension configuration.
use diffusion_types::{DiffusionError, Result};
use image::{DynamicImage, GrayImage, Luma};
use candle_core::Tensor;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};
pub trait Annotator: Send + Sync {
    fn annotate(&self, image: &DynamicImage) -> Result<DynamicImage>;
}
#[derive(Debug, Default, Clone, Copy)]
pub struct Canny;
impl Annotator for Canny {
    fn annotate(&self, image: &DynamicImage) -> Result<DynamicImage> {
        let gray = image.to_luma8();
        let mut out = GrayImage::new(gray.width(), gray.height());
        for y in 1..gray.height().saturating_sub(1) {
            for x in 1..gray.width().saturating_sub(1) {
                let gx = i32::from(gray.get_pixel(x + 1, y)[0]) - i32::from(gray.get_pixel(x - 1, y)[0]);
                let gy = i32::from(gray.get_pixel(x, y + 1)[0]) - i32::from(gray.get_pixel(x, y - 1)[0]);
                let v = ((gx.abs() + gy.abs()).min(255)) as u8;
                out.put_pixel(x, y, Luma([if v > 80 { 255 } else { 0 }]));
            }
        }
        Ok(DynamicImage::ImageLuma8(out))
    }
}
pub fn unavailable(name: &str) -> Result<()> {
    Err(DiffusionError::Model(format!("annotator `{name}` requires an external Candle model")))
}

#[derive(Debug, Clone)]
pub struct LoraTensorPair {
    pub target: String,
    pub down: Tensor,
    pub up: Tensor,
    pub alpha: Option<f32>,
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn load_lora(path: &Path, device: &candle_core::Device) -> Result<Vec<LoraTensorPair>> {
    let tensors = candle_core::safetensors::load(path, device).map_err(|e| DiffusionError::Model(format!("LoRA {}: {e}", path.display())))?;
    let mut down = BTreeMap::new();
    let mut up = BTreeMap::new();
    let mut alpha = BTreeMap::new();
    for (name, tensor) in tensors {
        let name = name.as_str();
        if let Some(target) = name.strip_suffix(".lora_down.weight").or_else(|| name.strip_suffix(".lora_A.weight")) {
            down.insert(target.to_owned(), tensor);
        } else if let Some(target) = name.strip_suffix(".lora_up.weight").or_else(|| name.strip_suffix(".lora_B.weight")) {
            up.insert(target.to_owned(), tensor);
        } else if let Some(target) = name.strip_suffix(".alpha") {
            let value = tensor.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| DiffusionError::Model(e.to_string()))?.first().copied().unwrap_or(1.0);
            alpha.insert(target.to_owned(), value);
        }
    }
    let mut result = Vec::new();
    for (target, down) in down {
        let up = up.remove(&target).ok_or_else(|| DiffusionError::Model(format!("LoRA missing up tensor for `{target}`")))?;
        result.push(LoraTensorPair { alpha: alpha.get(&target).copied(), target, down, up });
    }
    if result.is_empty() { return Err(DiffusionError::Model("LoRA contains no recognized tensor pairs".into())); }
    Ok(result)
}

pub fn load_conditioning(path: &Path, device: &candle_core::Device) -> Result<BTreeMap<String, Tensor>> {
    let tensors = candle_core::safetensors::load(path, device).map_err(|e| DiffusionError::Model(format!("conditioning {}: {e}", path.display())))?;
    let mut result = tensors.into_iter().collect::<BTreeMap<_, _>>();
    if !result.contains_key("prompt_embeds") || !result.contains_key("negative_prompt_embeds") {
        return Err(DiffusionError::Model("conditioning requires prompt_embeds and negative_prompt_embeds".into()));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage, Rgb};

    #[test]
    fn canny_detects_a_vertical_edge() {
        let mut image = RgbImage::new(8, 8);
        for y in 0..8 { for x in 4..8 { image.put_pixel(x, y, Rgb([255, 255, 255])); } }
        let output = Canny.annotate(&DynamicImage::ImageRgb8(image)).expect("canny").to_luma8();
        assert!(output.pixels().any(|pixel| pixel[0] == 255));
    }
}
