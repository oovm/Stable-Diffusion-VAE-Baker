//! Driver-only CLIP embedding primitives for Stable Diffusion 1.5.
//!
//! The tables remain in Titan device memory after construction. Encoding a
//! prompt performs both token and position lookup, followed by the residual
//! addition on the GPU; no Candle, Burn, CUDA Toolkit, or host-side tensor
//! execution is involved.

use crate::{F32Weight, load_f32_weight};
use std::path::Path;
use titan_hal::CudaContext;
use titan_tensor::CudaTensor;

const TEXT_CONTEXT: usize = 77;
const TEXT_HIDDEN: usize = 768;

/// CLIP embedding tables resident in NVIDIA driver-managed memory.
#[derive(Debug)]
pub struct TitanClipEmbeddings {
    token: CudaTensor,
    position: CudaTensor,
}

struct Linear {
    weight: CudaTensor,
    bias: CudaTensor,
}

impl Linear {
    fn load(path: &Path, prefix: &str, context: &CudaContext) -> Result<Self, String> {
        let weight = upload(load_f32_weight(path, &format!("{prefix}.weight")).map_err(|e| e.to_string())?, context.clone())?
            .transpose()
            .map_err(|e| format!("transpose {prefix}: {e:?}"))?;
        let bias = upload(load_f32_weight(path, &format!("{prefix}.bias")).map_err(|e| e.to_string())?, context.clone())?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        input.matmul(&self.weight).and_then(|x| x.add_bias(&self.bias)).map_err(|e| format!("linear: {e:?}"))
    }
}

struct ClipLayer {
    norm1_weight: CudaTensor,
    norm1_bias: CudaTensor,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm2_weight: CudaTensor,
    norm2_bias: CudaTensor,
    fc1: Linear,
    fc2: Linear,
}

impl ClipLayer {
    fn load(path: &Path, index: usize, context: &CudaContext) -> Result<Self, String> {
        let prefix = format!("text_model.encoder.layers.{index}");
        let weight = |name: &str| upload(load_f32_weight(path, name).map_err(|e| e.to_string())?, context.clone());
        Ok(Self {
            norm1_weight: weight(&format!("{prefix}.layer_norm1.weight"))?,
            norm1_bias: weight(&format!("{prefix}.layer_norm1.bias"))?,
            q: Linear::load(path, &format!("{prefix}.self_attn.q_proj"), context)?,
            k: Linear::load(path, &format!("{prefix}.self_attn.k_proj"), context)?,
            v: Linear::load(path, &format!("{prefix}.self_attn.v_proj"), context)?,
            out: Linear::load(path, &format!("{prefix}.self_attn.out_proj"), context)?,
            norm2_weight: weight(&format!("{prefix}.layer_norm2.weight"))?,
            norm2_bias: weight(&format!("{prefix}.layer_norm2.bias"))?,
            fc1: Linear::load(path, &format!("{prefix}.mlp.fc1"), context)?,
            fc2: Linear::load(path, &format!("{prefix}.mlp.fc2"), context)?,
        })
    }

    fn forward(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        let normalized =
            input.layer_norm(&self.norm1_weight, &self.norm1_bias, 1e-5).map_err(|e| format!("clip norm1: {e:?}"))?;
        let q = self.q.forward(&normalized)?;
        let k = self.k.forward(&normalized)?;
        let v = self.v.forward(&normalized)?;
        let mut heads = Vec::with_capacity(12);
        for head in 0..12 {
            let start = head * 64;
            heads.push(
                q.slice_columns(start, 64)
                    .map_err(|e| format!("q head: {e:?}"))?
                    .attention(
                        &k.slice_columns(start, 64).map_err(|e| format!("k head: {e:?}"))?,
                        &v.slice_columns(start, 64).map_err(|e| format!("v head: {e:?}"))?,
                    )
                    .map_err(|e| format!("attention: {e:?}"))?,
            );
        }
        let attention =
            CudaTensor::concat_columns(&heads.iter().collect::<Vec<_>>()).map_err(|e| format!("concat heads: {e:?}"))?;
        let residual = input.add(&self.out.forward(&attention)?).map_err(|e| format!("attention residual: {e:?}"))?;
        let normalized =
            residual.layer_norm(&self.norm2_weight, &self.norm2_bias, 1e-5).map_err(|e| format!("clip norm2: {e:?}"))?;
        let feed_forward = self.fc1.forward(&normalized)?.gelu().map_err(|e| format!("clip gelu: {e:?}"))?;
        residual.add(&self.fc2.forward(&feed_forward)?).map_err(|e| format!("mlp residual: {e:?}"))
    }
}

/// Complete SD 1.5 CLIP text encoder executed with Titan tensors.
pub struct TitanClipEncoder {
    embeddings: TitanClipEmbeddings,
    layers: Vec<ClipLayer>,
    final_weight: CudaTensor,
    final_bias: CudaTensor,
}

impl TitanClipEncoder {
    /// Loads all 12 transformer layers and their affine parameters.
    pub fn from_model_dir(model_dir: &Path, context: CudaContext) -> Result<Self, String> {
        let encoder = model_dir.join("text_encoder/model.safetensors");
        let embeddings = TitanClipEmbeddings::from_model_dir(model_dir, context.clone())?;
        let layers = (0..12).map(|index| ClipLayer::load(&encoder, index, &context)).collect::<Result<Vec<_>, _>>()?;
        let final_weight = upload(
            load_f32_weight(&encoder, "text_model.final_layer_norm.weight").map_err(|e| e.to_string())?,
            context.clone(),
        )?;
        let final_bias =
            upload(load_f32_weight(&encoder, "text_model.final_layer_norm.bias").map_err(|e| e.to_string())?, context)?;
        Ok(Self { embeddings, layers, final_weight, final_bias })
    }

    /// Runs CLIP and returns `[77, 768]` conditioning on the GPU.
    pub fn encode(&self, token_ids: &[usize]) -> Result<CudaTensor, String> {
        let mut hidden = self.embeddings.encode(token_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        hidden.layer_norm(&self.final_weight, &self.final_bias, 1e-5).map_err(|e| format!("clip final norm: {e:?}"))
    }
}

impl TitanClipEmbeddings {
    /// Loads the two Diffusers CLIP tables and uploads them to the GPU.
    pub fn from_model_dir(model_dir: &Path, context: CudaContext) -> Result<Self, String> {
        let encoder = model_dir.join("text_encoder/model.safetensors");
        let token = load_f32_weight(&encoder, "text_model.embeddings.token_embedding.weight").map_err(|e| e.to_string())?;
        let position =
            load_f32_weight(&encoder, "text_model.embeddings.position_embedding.weight").map_err(|e| e.to_string())?;
        validate_table(&token, "token embedding", 49_408)?;
        validate_table(&position, "position embedding", TEXT_CONTEXT)?;
        Ok(Self { token: upload(token, context.clone())?, position: upload(position, context)? })
    }

    /// Encodes exactly 77 token IDs into the CLIP conditioning matrix.
    pub fn encode(&self, token_ids: &[usize]) -> Result<CudaTensor, String> {
        if token_ids.len() != TEXT_CONTEXT {
            return Err(format!("CLIP requires {TEXT_CONTEXT} token IDs, got {}", token_ids.len()));
        }
        let token = self.token.gather_rows(token_ids).map_err(|e| format!("token gather: {e:?}"))?;
        let position_ids: Vec<usize> = (0..TEXT_CONTEXT).collect();
        let position = self.position.gather_rows(&position_ids).map_err(|e| format!("position gather: {e:?}"))?;
        token.add(&position).map_err(|e| format!("embedding add: {e:?}"))
    }

    /// Returns the hidden width of the encoder output.
    pub const fn hidden_size(&self) -> usize {
        TEXT_HIDDEN
    }
}

fn upload(weight: F32Weight, context: CudaContext) -> Result<CudaTensor, String> {
    CudaTensor::from_slice(context, weight.shape, &weight.values).map_err(|e| format!("upload {}: {e:?}", weight.name))
}

fn validate_table(weight: &F32Weight, label: &str, rows: usize) -> Result<(), String> {
    if weight.shape != vec![rows, TEXT_HIDDEN] {
        return Err(format!("{label} shape must be [{rows}, {TEXT_HIDDEN}], got {:?}", weight.shape));
    }
    Ok(())
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::open_titan_cuda;

    #[test]
    fn real_sd15_embeddings_encode_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let clip = TitanClipEmbeddings::from_model_dir(&model_dir, context).expect("CLIP tables");
        let ids: Vec<usize> = (0..TEXT_CONTEXT).map(|i| i % 10).collect();
        let output = clip.encode(&ids).expect("GPU embedding");
        assert_eq!(output.shape(), &[TEXT_CONTEXT, TEXT_HIDDEN]);
        assert_eq!(output.to_vec().expect("readback").len(), TEXT_CONTEXT * TEXT_HIDDEN);
    }

    #[test]
    fn real_sd15_clip_transformer_encode_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let clip = TitanClipEncoder::from_model_dir(&model_dir, context).expect("full CLIP weights");
        let ids: Vec<usize> = (0..TEXT_CONTEXT).map(|i| i % 10).collect();
        let output = clip.encode(&ids).expect("full GPU CLIP");
        assert_eq!(output.shape(), &[TEXT_CONTEXT, TEXT_HIDDEN]);
        assert!(output.to_vec().expect("readback").iter().all(|value| value.is_finite()));
    }
}
