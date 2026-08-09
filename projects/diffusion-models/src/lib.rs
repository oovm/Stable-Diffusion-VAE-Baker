//! Safe model inspection and format detection.
use diffusion_types::{DiffusionError, ModelFamily, Result};
use safetensors::{SafeTensors, tensor::Dtype};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, RwLock},
};


/// Minimal DDIM scheduler state for an SD 1.5 denoising run.
#[derive(Clone, Debug)]
pub struct DdimScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f32>,
}

impl DdimScheduler {
    /// Builds a linear-beta DDIM schedule with `steps` inference steps.
    pub fn new(steps: usize, train_steps: usize) -> Result<Self> {
        if steps == 0 || train_steps < 2 || steps > train_steps {
            return Err(DiffusionError::InvalidRequest("invalid DDIM step count".into()));
        }
        let mut alpha = 1.0_f32;
        let mut cumulative = Vec::with_capacity(train_steps);
        for index in 0..train_steps {
            let beta = 0.00085 + (0.012 - 0.00085) * index as f32 / (train_steps - 1) as f32;
            alpha *= 1.0 - beta;
            cumulative.push(alpha);
        }
        let timesteps = (0..steps).map(|index| train_steps - 1 - index * train_steps / steps).collect();
        Ok(Self { timesteps, alphas_cumprod: cumulative })
    }

    /// Returns descending training indices used by the denoiser.
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Applies the deterministic DDIM (`eta=0`) update elementwise on host values.
    pub fn step(&self, sample: &[f32], noise: &[f32], step: usize) -> Result<Vec<f32>> {
        if sample.len() != noise.len() || step >= self.timesteps.len() {
            return Err(DiffusionError::InvalidRequest("DDIM sample/noise shape mismatch".into()));
        }
        let timestep = self.timesteps[step];
        let previous = if step + 1 < self.timesteps.len() { self.timesteps[step + 1] } else { 0 };
        let alpha = self.alphas_cumprod[timestep];
        let previous_alpha = self.alphas_cumprod[previous];
        let sqrt_alpha = alpha.sqrt();
        let sqrt_one_minus = (1.0 - alpha).sqrt();
        let sqrt_previous = previous_alpha.sqrt();
        let direction_scale = (1.0 - previous_alpha).sqrt();
        Ok(sample
            .iter()
            .zip(noise)
            .map(|(sample, noise)| {
                let predicted = (sample - noise * sqrt_one_minus) / sqrt_alpha;
                predicted * sqrt_previous + noise * direction_scale
            })
            .collect())
    }
}
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub family: ModelFamily,
    pub derivative: Option<String>,
    pub tensors: usize,
    pub keys: BTreeSet<String>,
    pub components: BTreeMap<String, usize>,
    pub metadata: BTreeMap<String, String>,
}
pub fn inspect(path: &Path) -> Result<ModelInfo> {
    let file = if path.is_dir() { path.join("model.safetensors") } else { path.to_path_buf() };
    let bytes = fs::read(&file).map_err(|e| DiffusionError::Model(format!("{}: {e}", file.display())))?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| DiffusionError::Model(e.to_string()))?;
    let keys: BTreeSet<_> = st.names().into_iter().map(|key| key.to_owned()).collect();
    let family = if keys.iter().any(|k| k.contains("conditioner.embedders.1") || k.contains("conditioner.embedders.1.model")) {
        ModelFamily::Sdxl
    }
    else if keys.iter().any(|k| k.contains("model.diffusion_model.input_blocks.0.0.weight"))
        && keys.iter().any(|k| k.contains("input_blocks.0.0.weight"))
    {
        ModelFamily::Sd15
    }
    else {
        ModelFamily::Sd15
    };
    let derivative = if keys.iter().any(|k| k.to_ascii_lowercase().contains("pony")) { Some("pony".into()) } else { None };
    let mut components = BTreeMap::new();
    for key in &keys {
        let component = if key.starts_with("model.diffusion_model.") || key.starts_with("unet.") {
            "unet"
        }
        else if key.starts_with("first_stage_model.") || key.starts_with("vae.") {
            "vae"
        }
        else if key.starts_with("cond_stage_model.") || key.starts_with("conditioner.") {
            "text_encoder"
        }
        else {
            "other"
        };
        *components.entry(component.into()).or_insert(0) += 1;
    }
    let metadata = BTreeMap::new();
    Ok(ModelInfo { family, derivative, tensors: keys.len(), keys, components, metadata })
}

/// A contiguous Diffusers weight converted to f32 for Titan execution.
#[derive(Clone, Debug)]
pub struct F32Weight {
    /// Original tensor name in the safetensors file.
    pub name: String,
    /// Row-major tensor dimensions.
    pub shape: Vec<usize>,
    /// Converted f32 values.
    pub values: Vec<f32>,
}

// Reuse the immutable safetensors file buffer while assembling a native
// pipeline. Re-reading a multi-gigabyte file for every tensor exhausts Windows
// process resources before the complete SD15 UNet can be constructed.
fn safetensor_file_cache() -> &'static RwLock<std::collections::HashMap<PathBuf, Arc<Vec<u8>>>> {
    static CACHE: OnceLock<RwLock<std::collections::HashMap<PathBuf, Arc<Vec<u8>>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(std::collections::HashMap::new()))
}

fn cached_safetensor_bytes(path: &Path) -> Result<Arc<Vec<u8>>> {
    if let Some(bytes) = safetensor_file_cache().read().expect("safetensor cache poisoned").get(path).cloned() {
        return Ok(bytes);
    }
    let bytes = Arc::new(fs::read(path).map_err(|error| DiffusionError::Model(format!("{}: {error}", path.display())))?);
    let mut cache = safetensor_file_cache().write().expect("safetensor cache poisoned");
    Ok(cache.entry(path.to_path_buf()).or_insert_with(|| bytes.clone()).clone())
}

/// Loads one F32, F16, or BF16 safetensors tensor and validates its byte size.
pub fn load_f32_weight(path: &Path, name: &str) -> Result<F32Weight> {
    let bytes = cached_safetensor_bytes(path)?;
    let tensors = SafeTensors::deserialize(bytes.as_slice()).map_err(|error| DiffusionError::Model(error.to_string()))?;
    let tensor = tensors.tensor(name).map_err(|error| DiffusionError::Model(format!("{name}: {error}")))?;
    let count = tensor
        .shape()
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| DiffusionError::Model(format!("{name}: shape element count overflow")))?;
    let data = tensor.data();
    let values = match tensor.dtype() {
        Dtype::F32 => {
            if data.len() != count * 4 {
                return Err(DiffusionError::Model(format!("{name}: invalid F32 byte length")));
            }
            data.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32"))).collect()
        }
        Dtype::F16 => {
            if data.len() != count * 2 {
                return Err(DiffusionError::Model(format!("{name}: invalid F16 byte length")));
            }
            data.chunks_exact(2)
                .map(|chunk| half::f16::from_bits(u16::from_le_bytes(chunk.try_into().expect("two-byte f16"))).to_f32())
                .collect()
        }
        Dtype::BF16 => {
            if data.len() != count * 2 {
                return Err(DiffusionError::Model(format!("{name}: invalid BF16 byte length")));
            }
            data.chunks_exact(2)
                .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().expect("two-byte bf16"))).to_f32())
                .collect()
        }
        dtype => return Err(DiffusionError::Model(format!("{name}: unsupported tensor dtype {dtype:?}"))),
    };
    Ok(F32Weight { name: name.into(), shape: tensor.shape().to_vec(), values })
}

/// Loads a complete SD 1.5 CLIP token embedding table for Titan upload.
pub fn load_sd15_token_embedding(model_dir: &Path) -> Result<F32Weight> {
    load_f32_weight(&model_dir.join("text_encoder/model.safetensors"), "text_model.embeddings.token_embedding.weight")
}


#[cfg(test)]
mod scheduler_tests {
    use super::*;
    #[test]
    fn ddim_schedule_is_descending_and_deterministic() {
        let scheduler = DdimScheduler::new(4, 1000).expect("schedule");
        assert_eq!(scheduler.timesteps(), &[999, 749, 499, 249]);
        let first = scheduler.step(&[0.5, -0.25], &[0.1, 0.2], 0).expect("step");
        let second = scheduler.step(&[0.5, -0.25], &[0.1, 0.2], 0).expect("step");
        assert_eq!(first, second);
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;
    use safetensors::tensor::{TensorView, serialize};
    #[test]
    fn loads_f32_weight_with_shape_validation() {
        let path = std::env::temp_dir().join(format!("titan-weight-{}.safetensors", std::process::id()));
        let values = [1.0_f32, -2.0];
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![2], &bytes).expect("view");
        fs::write(&path, serialize([("weight", view)].into_iter(), None).expect("serialize")).expect("write");
        let weight = load_f32_weight(&path, "weight").expect("load");
        assert_eq!(weight.shape, vec![2]);
        assert_eq!(weight.values, values);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn loads_the_real_sd15_clip_embedding_table() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let weight = load_sd15_token_embedding(&path).expect("real SD15 CLIP embedding");
        assert_eq!(weight.shape, vec![49_408, 768]);
        assert_eq!(weight.values.len(), 49_408 * 768);
    }

}
