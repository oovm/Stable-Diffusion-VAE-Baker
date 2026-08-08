//! Safe model inspection and format detection.
use diffusion_types::{DevicePreference, DiffusionError, ModelFamily, Result};
use safetensors::{SafeTensors, tensor::Dtype};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

/// Titan Driver API implementation of the SD 1.5 CLIP embedding stage.
pub mod titan_clip;

/// Opens a CUDA Driver API session through the Titan Git dependency.
///
/// This is deliberately a driver-only capability probe. Model execution is
/// added only after Titan publishes device buffers and kernel dispatch.
pub fn open_titan_cuda(ordinal: usize) -> std::result::Result<titan_hal::CudaDriver, titan_hal::CudaDriverError> {
    titan_hal::CudaDriver::open(ordinal)
}

/// The actual execution target selected for the native Titan path.
#[derive(Debug)]
pub enum TitanDevice {
    /// Host execution is selected explicitly or after an automatic fallback.
    Cpu,
    /// NVIDIA Driver API execution has an active primary CUDA context.
    Cuda(titan_hal::CudaContext),
}

/// Selects the native Titan execution target for one `sd.exe` invocation.
///
/// `auto` attempts the NVIDIA Driver API and falls back to CPU. `cuda`
/// propagates the Driver API error, so a requested GPU can never silently run
/// the model on the CPU.
pub fn select_titan_device(preference: DevicePreference) -> std::result::Result<TitanDevice, titan_hal::CudaDriverError> {
    match preference {
        DevicePreference::Cpu => Ok(TitanDevice::Cpu),
        DevicePreference::Cuda => Ok(TitanDevice::Cuda(open_titan_cuda(0)?.primary_context()?)),
        DevicePreference::Auto => match open_titan_cuda(0).and_then(|driver| driver.primary_context()) {
            Ok(context) => Ok(TitanDevice::Cuda(context)),
            Err(_) => Ok(TitanDevice::Cpu),
        },
    }
}

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

/// Loads one F32, F16, or BF16 safetensors tensor and validates its byte size.
pub fn load_f32_weight(path: &Path, name: &str) -> Result<F32Weight> {
    let bytes = fs::read(path).map_err(|error| DiffusionError::Model(format!("{}: {error}", path.display())))?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|error| DiffusionError::Model(error.to_string()))?;
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

/// Uploads a decoded weight into a Titan Driver API tensor.
pub fn upload_weight(weight: &F32Weight, context: titan_hal::CudaContext) -> Result<titan_tensor::CudaTensor> {
    titan_tensor::CudaTensor::from_slice(context, weight.shape.clone(), &weight.values)
        .map_err(|error| DiffusionError::Model(format!("upload {}: {error:?}", weight.name)))
}

#[cfg(all(test, windows))]
mod titan_integration_tests {
    #[test]
    fn opens_the_nvidia_driver_from_the_remote_titan_dependency() {
        let driver = super::open_titan_cuda(0).expect("NVIDIA driver device 0");
        assert!(driver.device_count() >= 1);
    }

    #[test]
    fn dispatches_a_tensor_addition_from_the_remote_titan_dependency() {
        let context = super::open_titan_cuda(0).expect("NVIDIA driver device 0").primary_context().expect("CUDA context");
        let left = titan_tensor::CudaTensor::from_slice(context.clone(), vec![2], &[1.0, 2.0]).expect("left upload");
        let right = titan_tensor::CudaTensor::from_slice(context, vec![2], &[0.5, 1.5]).expect("right upload");
        assert_eq!(left.add(&right).expect("CUDA add").to_vec().expect("result download"), vec![1.5, 3.5]);
    }

    #[test]
    fn dispatches_a_matrix_multiplication_from_the_remote_titan_dependency() {
        let context = super::open_titan_cuda(0).expect("NVIDIA driver device 0").primary_context().expect("CUDA context");
        let left = titan_tensor::CudaTensor::from_slice(context.clone(), vec![1, 2], &[2.0, 3.0]).expect("left upload");
        let right = titan_tensor::CudaTensor::from_slice(context, vec![2, 1], &[4.0, 5.0]).expect("right upload");
        assert_eq!(left.matmul(&right).expect("CUDA matmul").to_vec().expect("result download"), vec![23.0]);
    }

    #[test]
    fn auto_and_cuda_select_an_active_titan_cuda_context() {
        assert!(matches!(super::select_titan_device(diffusion_types::DevicePreference::Auto), Ok(super::TitanDevice::Cuda(_))));
        assert!(matches!(super::select_titan_device(diffusion_types::DevicePreference::Cuda), Ok(super::TitanDevice::Cuda(_))));
        assert!(matches!(super::select_titan_device(diffusion_types::DevicePreference::Cpu), Ok(super::TitanDevice::Cpu)));
    }
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

    #[cfg(windows)]
    #[test]
    fn uploads_and_reads_back_the_real_sd15_clip_embedding_prefix() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let weight = load_sd15_token_embedding(&path).expect("real SD15 CLIP embedding");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("CUDA context");
        let tensor = upload_weight(&weight, context).expect("upload embedding");
        let prefix = tensor.gather_rows(&[0, 1]).expect("embedding prefix").to_vec().expect("download prefix");
        assert_eq!(prefix.len(), 2 * 768);
        assert_eq!(prefix[0], weight.values[0]);
        assert_eq!(prefix[768], weight.values[768]);
    }
}
