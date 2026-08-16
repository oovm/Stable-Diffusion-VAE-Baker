//! Safe model inspection and format detection.
use diffusion_types::{DiffusionError, ModelFamily, Result};
use safetensors::{SafeTensors, tensor::Dtype};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, RwLock},
};
use titan_backend_cpu::CpuDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device as TitanDevice, F32Tensor, TensorHandle};
use titan_types::{AliasContract, AttrMap, AttrValue, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};
use tokenizers::Tokenizer;


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
    /// Source safetensors dtype before conversion.
    pub source_dtype: Dtype,
    /// Converted f32 values.
    pub values: Vec<f32>,
}

/// Fixed-length SD 1.5 CLIP token IDs ready for a backend upload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sd15PromptTokens {
    /// Exactly 77 IDs, including tokenizer-owned special tokens and EOT padding.
    pub ids: Vec<usize>,
}

/// Opaque CPU bridge backed by Titan's backend-neutral tensor contract.
pub struct TitanCpuBridge {
    device: TitanDevice,
}

/// A one-dimensional F32 tensor owned by Titan's runtime device.
pub struct TitanCpuTensor1 {
    tensor: F32Tensor<1>,
}

/// A dynamic `[77, 768]` tensor retained through Titan's opaque handle contract.
pub struct TitanCpuTensor2 {
    handle: TensorHandle,
}

impl TitanCpuBridge {
    /// Opens Titan's portable CPU device (ordinal zero).
    pub fn open() -> Result<Self> {
        let driver = CpuDriver;
        let fingerprint = driver
            .enumerate()
            .map_err(|error| DiffusionError::Model(format!("Titan CPU enumerate failed: {error}")))?
            .into_iter()
            .next()
            .ok_or_else(|| DiffusionError::Model("Titan CPU has no devices".into()))?;
        let session = driver
            .open(fingerprint.device)
            .map_err(|error| DiffusionError::Model(format!("Titan CPU open failed: {error}")))?;
        Ok(Self { device: TitanDevice::from_session(session) })
    }

    /// Uploads a contiguous weight slice without retaining the source file buffer.
    pub fn upload_weight_slice(&self, weight: &F32Weight, range: std::ops::Range<usize>) -> Result<TitanCpuTensor1> {
        let values = weight
            .values
            .get(range)
            .ok_or_else(|| DiffusionError::InvalidRequest(format!("{}: weight slice is out of bounds", weight.name)))?;
        self.upload_f32(values)
    }

    /// Uploads fixed-length tokenizer IDs using lossless F32 representation.
    pub fn upload_prompt_tokens(&self, tokens: &Sd15PromptTokens) -> Result<TitanCpuTensor1> {
        let values: Vec<f32> = tokens.ids.iter().map(|id| *id as f32).collect();
        self.upload_f32(&values)
    }

    /// Uploads a host vector for a backend-neutral Titan operation.
    pub fn upload_values(&self, values: &[f32]) -> Result<TitanCpuTensor1> {
        self.upload_f32(values)
    }

    /// Uploads the SD1.5 CLIP sequence embedding with its semantic rank preserved.
    pub fn upload_clip_sequence(&self, values: &[f32]) -> Result<TitanCpuTensor2> {
        if values.len() != 77 * 768 {
            return Err(DiffusionError::InvalidRequest("SD15 CLIP sequence must contain [77, 768] values".into()));
        }
        let handle = TensorHandle::from_f32_vec(self.device.session().clone(), vec![77, 768], values)
            .map_err(|error| DiffusionError::Model(format!("Titan CPU sequence upload failed: {error}")))?;
        Ok(TitanCpuTensor2 { handle })
    }

    /// Executes Titan Runtime's CPU LayerNorm contract with learned gamma/beta.
    pub fn layer_norm_readback(
        &self,
        input: &TitanCpuTensor2,
        gamma: &TitanCpuTensor1,
        beta: &TitanCpuTensor1,
    ) -> Result<Vec<f32>> {
        let mut attrs = AttrMap::new();
        attrs.insert("epsilon".into(), AttrValue::Float((1e-5_f64).to_bits()));
        let request = OpRequest {
            operator: OperatorId("layer_norm".into()),
            inputs: vec![input.handle.clone(), gamma.tensor.handle(), beta.tensor.handle()],
            outputs: vec![TensorSpec {
                dtype: DType::F32,
                shape: Shape(vec![77, 768]),
                strides: Strides(vec![768, 1]),
                layout: Layout::Contiguous,
                alias: AliasContract::NoAlias,
            }],
            attrs,
            effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
            source: SourceSpan { file: "diffusion-models::TitanCpuBridge::layer_norm_readback".into(), line: 1, column: 1 },
        };
        let mut runtime = Runtime::open(std::env::temp_dir().join("stable-native-sd15-titan-cpu"));
        let output = runtime
            .execute(request)
            .map_err(|error| DiffusionError::Model(format!("Titan CPU LayerNorm dispatch failed: {error}")))?
            .wait()
            .map_err(|error| DiffusionError::Model(format!("Titan CPU LayerNorm wait failed: {error}")))?
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| DiffusionError::Model("Titan CPU LayerNorm returned no output".into()))?;
        output
            .to_vec_f32()
            .map_err(|error| DiffusionError::Model(format!("Titan CPU LayerNorm output read failed: {error}")))
    }

    /// Adds two same-shaped tensors through Titan Runtime's generated CPU contract.
    pub fn add_readback(&self, left: &TitanCpuTensor1, right: &TitanCpuTensor1) -> Result<Vec<f32>> {
        let left_handle = left.tensor.handle();
        let right_handle = right.tensor.handle();
        if left_handle.shape() != right_handle.shape() {
            return Err(DiffusionError::InvalidRequest("Titan CPU add shape mismatch".into()));
        }
        let shape = left_handle.shape().iter().map(|dimension| *dimension as u64).collect::<Vec<_>>();
        let mut attrs = AttrMap::new();
        attrs.insert("operation".into(), AttrValue::String("add".into()));
        let request = OpRequest {
            operator: OperatorId("elementwise.fused".into()),
            inputs: vec![left_handle, right_handle],
            outputs: vec![TensorSpec {
                dtype: DType::F32,
                shape: Shape(shape),
                strides: Strides(vec![1]),
                layout: Layout::Contiguous,
                alias: AliasContract::NoAlias,
            }],
            attrs,
            effects: EffectContract { memory: MemoryEffect::Pure, deterministic: true },
            source: SourceSpan { file: "diffusion-models::TitanCpuBridge::add_readback".into(), line: 1, column: 1 },
        };
        let mut runtime = Runtime::open(std::env::temp_dir().join("stable-native-sd15-titan-cpu"));
        let output = runtime
            .execute(request)
            .map_err(|error| DiffusionError::Model(format!("Titan CPU add dispatch failed: {error}")))?
            .wait()
            .map_err(|error| DiffusionError::Model(format!("Titan CPU add wait failed: {error}")))?
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| DiffusionError::Model("Titan CPU add returned no output".into()))?;
        output
            .to_vec_f32()
            .map_err(|error| DiffusionError::Model(format!("Titan CPU add output read failed: {error}")))
    }

    fn upload_f32(&self, values: &[f32]) -> Result<TitanCpuTensor1> {
        let tensor = F32Tensor::from_slice(&self.device, [values.len()], values)
            .map_err(|error| DiffusionError::Model(format!("Titan CPU upload failed: {error}")))?;
        Ok(TitanCpuTensor1 { tensor })
    }
}

impl TitanCpuTensor1 {
    /// Synchronizes the Titan tensor and downloads its F32 values.
    pub fn read_f32(&self) -> Result<Vec<f32>> {
        self.tensor
            .to_vec()
            .map_err(|error| DiffusionError::Model(format!("Titan CPU readback failed: {error}")))
    }
}

/// Encodes one prompt with the repository's SD 1.5 CLIP tokenizer.
pub fn tokenize_sd15_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Sd15PromptTokens> {
    const CONTEXT: usize = 77;
    let mut ids = tokenizer
        .encode(prompt, true)
        .map_err(|error| DiffusionError::Model(format!("tokenizer encode failed: {error}")))?
        .get_ids()
        .to_vec();
    if ids.len() > CONTEXT {
        return Err(DiffusionError::InvalidRequest(format!("prompt exceeds {CONTEXT} CLIP tokens")));
    }
    let pad = tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .copied()
        .ok_or_else(|| DiffusionError::Model("SD 1.5 tokenizer has no <|endoftext|> token".into()))?;
    ids.resize(CONTEXT, pad);
    Ok(Sd15PromptTokens { ids: ids.into_iter().map(|id| id as usize).collect() })
}

/// Loads one tensor from a Diffusers component directory.
pub fn load_diffusers_weight(model_dir: &Path, component: &str, name: &str) -> Result<F32Weight> {
    if component.contains('/') || component.contains('\\') || component == "." || component == ".." {
        return Err(DiffusionError::InvalidRequest(format!("invalid Diffusers component: {component}")));
    }
    load_f32_weight(&model_dir.join(component).join("model.safetensors"), name)
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
    let source_dtype = tensor.dtype();
    let values = match source_dtype {
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
    Ok(F32Weight { name: name.into(), shape: tensor.shape().to_vec(), source_dtype, values })
}

/// Loads a complete SD 1.5 CLIP token embedding table for Titan upload.
pub fn load_sd15_token_embedding(model_dir: &Path) -> Result<F32Weight> {
    load_diffusers_weight(model_dir, "text_encoder", "text_model.embeddings.token_embedding.weight")
}

/// Loads SD 1.5 CLIP's learned 77-position embedding table.
pub fn load_sd15_position_embedding(model_dir: &Path) -> Result<F32Weight> {
    load_diffusers_weight(model_dir, "text_encoder", "text_model.embeddings.position_embedding.weight")
}

/// Performs the embedding-table lookup portion of the CLIP input path.
pub fn lookup_sd15_token_embeddings(tokens: &Sd15PromptTokens, table: &F32Weight) -> Result<Vec<f32>> {
    if table.shape != [49_408, 768] {
        return Err(DiffusionError::Model(format!("{}: expected SD15 token table [49408, 768]", table.name)));
    }
    if tokens.ids.len() != 77 {
        return Err(DiffusionError::InvalidRequest("SD15 token input must contain 77 IDs".into()));
    }
    let mut output = Vec::with_capacity(tokens.ids.len() * table.shape[1]);
    for &token in &tokens.ids {
        let start = token
            .checked_mul(table.shape[1])
            .ok_or_else(|| DiffusionError::InvalidRequest("token embedding offset overflow".into()))?;
        let row = table
            .values
            .get(start..start + table.shape[1])
            .ok_or_else(|| DiffusionError::InvalidRequest(format!("token ID {token} is outside the SD15 vocabulary")))?;
        output.extend_from_slice(row);
    }
    Ok(output)
}

/// Returns the learned position rows in row-major layout for the 77-token input.
pub fn sd15_position_values(tokens: &Sd15PromptTokens, table: &F32Weight) -> Result<Vec<f32>> {
    if table.shape != [77, 768] {
        return Err(DiffusionError::Model(format!("{}: expected SD15 position table [77, 768]", table.name)));
    }
    if tokens.ids.len() != 77 {
        return Err(DiffusionError::InvalidRequest("SD15 token input must contain 77 IDs".into()));
    }
    Ok(table.values.clone())
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
        assert_eq!(weight.source_dtype, Dtype::F32);
        assert_eq!(weight.values, values);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn loads_the_real_sd15_clip_embedding_table() {
        let path = std::env::var_os("SD15_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\AI 生图\stable-diffusion.rs\models\sd15"));
        let weight = load_sd15_token_embedding(&path).expect("real SD15 CLIP embedding");
        assert_eq!(weight.shape, vec![49_408, 768]);
        assert_eq!(weight.source_dtype, Dtype::F32);
        assert_eq!(weight.values.len(), 49_408 * 768);
        assert_eq!(weight.values[0].to_bits(), 0xba9d_ebb0);
    }

    #[test]
    fn uploads_and_reads_back_a_real_weight_slice_on_titan_cpu() {
        let path = std::env::var_os("SD15_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\AI 生图\stable-diffusion.rs\models\sd15"));
        let weight = load_sd15_token_embedding(&path).expect("real SD15 CLIP embedding");
        let expected = weight.values[..8].to_vec();
        let bridge = TitanCpuBridge::open().expect("Titan CPU");
        let tensor = bridge.upload_weight_slice(&weight, 0..expected.len()).expect("upload weight slice");
        assert_eq!(tensor.read_f32().expect("read weight slice"), expected);
    }

}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;

    #[test]
    fn encodes_real_sd15_prompt_to_fixed_token_ids() {
        let path = std::env::var_os("SD15_TOKENIZER")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\AI 生图\stable-diffusion.rs\models\sd15\tokenizer\tokenizer.json"));
        let tokenizer = Tokenizer::from_file(&path).expect("real SD15 tokenizer");
        let tokens = tokenize_sd15_prompt(&tokenizer, "a photo of a cat").expect("prompt tokens");
        assert_eq!(tokens.ids.len(), 77);
        assert_eq!(&tokens.ids[..8], &[49406, 320, 1125, 539, 320, 2368, 49407, 49407]);
        assert!(tokens.ids[7..].iter().all(|id| *id == 49407));
        let bridge = TitanCpuBridge::open().expect("Titan CPU");
        let tensor = bridge.upload_prompt_tokens(&tokens).expect("upload prompt IDs");
        let round_trip = tensor.read_f32().expect("read prompt IDs");
        assert_eq!(round_trip.len(), tokens.ids.len());
        for (actual, expected) in round_trip.iter().zip(&tokens.ids) {
            assert_eq!(*actual, *expected as f32);
        }
    }

    #[test]
    fn executes_sd15_embedding_lookup_plus_position_add_on_titan_cpu() {
        let model_dir = std::env::var_os("SD15_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\AI 生图\stable-diffusion.rs\models\sd15"));
        let tokenizer_path = std::env::var_os("SD15_TOKENIZER")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_dir.join("tokenizer/tokenizer.json"));
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("real SD15 tokenizer");
        let tokens = tokenize_sd15_prompt(&tokenizer, "a photo of a cat").expect("prompt tokens");
        let token_table = load_sd15_token_embedding(&model_dir).expect("real token embedding");
        let position_table = load_sd15_position_embedding(&model_dir).expect("real position embedding");
        let token_rows = lookup_sd15_token_embeddings(&tokens, &token_table).expect("embedding lookup");
        let position_rows = sd15_position_values(&tokens, &position_table).expect("position rows");
        let expected: Vec<f32> = token_rows.iter().zip(&position_rows).map(|(token, position)| token + position).collect();

        let bridge = TitanCpuBridge::open().expect("Titan CPU");
        let token_tensor = bridge.upload_values(&token_rows).expect("upload token rows");
        let position_tensor = bridge.upload_values(&position_rows).expect("upload position rows");
        let actual = bridge.add_readback(&token_tensor, &position_tensor).expect("Titan CPU position add");
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-7, "Titan add mismatch: {actual} vs {expected}");
        }
    }

    #[test]
    fn executes_sd15_first_layer_norm_on_titan_cpu() {
        let model_dir = std::env::var_os("SD15_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\AI 生图\stable-diffusion.rs\models\sd15"));
        let tokenizer_path = std::env::var_os("SD15_TOKENIZER")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_dir.join("tokenizer/tokenizer.json"));
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("real SD15 tokenizer");
        let tokens = tokenize_sd15_prompt(&tokenizer, "a photo of a cat").expect("prompt tokens");
        let token_table = load_sd15_token_embedding(&model_dir).expect("real token embedding");
        let position_table = load_sd15_position_embedding(&model_dir).expect("real position embedding");
        let token_rows = lookup_sd15_token_embeddings(&tokens, &token_table).expect("embedding lookup");
        let position_rows = sd15_position_values(&tokens, &position_table).expect("position rows");
        let gamma = load_f32_weight(
            &model_dir.join("text_encoder/model.safetensors"),
            "text_model.encoder.layers.0.layer_norm1.weight",
        )
        .expect("real first LayerNorm gamma");
        let beta = load_f32_weight(
            &model_dir.join("text_encoder/model.safetensors"),
            "text_model.encoder.layers.0.layer_norm1.bias",
        )
        .expect("real first LayerNorm beta");
        assert_eq!(gamma.shape, vec![768]);
        assert_eq!(beta.shape, vec![768]);

        let bridge = TitanCpuBridge::open().expect("Titan CPU");
        let input = bridge
            .add_readback(
                &bridge.upload_values(&token_rows).expect("upload token rows"),
                &bridge.upload_values(&position_rows).expect("upload position rows"),
            )
            .expect("Titan CPU embedding add");
        let actual = bridge
            .layer_norm_readback(
                &bridge.upload_clip_sequence(&input).expect("upload CLIP sequence"),
                &bridge.upload_values(&gamma.values).expect("upload gamma"),
                &bridge.upload_values(&beta.values).expect("upload beta"),
            )
            .expect("Titan CPU first LayerNorm");
        let expected = cpu_layer_norm_reference(&input, &gamma.values, &beta.values, 1e-5);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "Titan LayerNorm mismatch: {actual} vs {expected}");
        }
    }

    fn cpu_layer_norm_reference(input: &[f32], gamma: &[f32], beta: &[f32], epsilon: f64) -> Vec<f32> {
        input
            .chunks_exact(768)
            .flat_map(|row| {
                let mean = row.iter().map(|value| *value as f64).sum::<f64>() / 768.0;
                let variance = row.iter().map(|value| (*value as f64 - mean).powi(2)).sum::<f64>() / 768.0;
                let inverse_stddev = 1.0 / (variance + epsilon).sqrt();
                row.iter()
                    .zip(gamma)
                    .zip(beta)
                    .map(|((value, gamma), beta)| (((*value as f64 - mean) * inverse_stddev) as f32 * *gamma) + *beta)
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}
