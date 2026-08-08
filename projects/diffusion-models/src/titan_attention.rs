//! Diffusers spatial transformer blocks executed on Titan tensors.

use crate::{F32Weight, load_f32_weight};
use std::path::Path;
use titan_hal::CudaContext;
use titan_tensor::CudaTensor;

fn upload(weight: F32Weight, context: &CudaContext) -> Result<CudaTensor, String> {
    CudaTensor::from_slice(context.clone(), weight.shape, &weight.values).map_err(|e| format!("upload {}: {e:?}", weight.name))
}

fn load(path: &Path, name: &str, context: &CudaContext) -> Result<CudaTensor, String> {
    upload(load_f32_weight(path, name).map_err(|e| e.to_string())?, context)
}

struct Linear {
    weight: CudaTensor,
    bias: Option<CudaTensor>,
}

impl Linear {
    fn load(path: &Path, prefix: &str, has_bias: bool, context: &CudaContext) -> Result<Self, String> {
        let raw = crate::load_f32_weight(path, &format!("{prefix}.weight")).map_err(|e| e.to_string())?;
        if raw.shape.len() != 2 {
            return Err(format!("{prefix}.weight must be rank 2, got {:?}", raw.shape));
        }
        let [out, input] = [raw.shape[0], raw.shape[1]];
        let weight =
            upload(F32Weight { name: raw.name, shape: vec![input, out], values: transpose(&raw.values, out, input) }, context)?;
        let bias = has_bias.then(|| load(path, &format!("{prefix}.bias"), context)).transpose()?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        let output = input.matmul(&self.weight).map_err(|e| format!("linear: {e:?}"))?;
        match &self.bias {
            Some(bias) => output.add_bias(bias).map_err(|e| format!("linear bias: {e:?}")),
            None => Ok(output),
        }
    }
}

struct Norm {
    weight: CudaTensor,
    bias: CudaTensor,
}

impl Norm {
    fn load(path: &Path, prefix: &str, context: &CudaContext) -> Result<Self, String> {
        Ok(Self {
            weight: load(path, &format!("{prefix}.weight"), context)?,
            bias: load(path, &format!("{prefix}.bias"), context)?,
        })
    }
    fn forward(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        input.layer_norm(&self.weight, &self.bias, 1e-5).map_err(|e| format!("transformer norm: {e:?}"))
    }
}

/// One Diffusers `BasicTransformerBlock` with self-attention, cross-attention,
/// and GEGLU feed-forward residuals.
pub struct TitanBasicTransformerBlock {
    norm1: Norm,
    self_q: Linear,
    self_k: Linear,
    self_v: Linear,
    self_out: Linear,
    norm2: Norm,
    cross_q: Linear,
    cross_k: Linear,
    cross_v: Linear,
    cross_out: Linear,
    norm3: Norm,
    ff_in: Linear,
    ff_out: Linear,
}

impl TitanBasicTransformerBlock {
    /// Loads a block below a Diffusers transformer prefix.
    pub fn from_model(model_dir: &Path, prefix: &str, context: &CudaContext) -> Result<Self, String> {
        let path = model_dir.join("unet/diffusion_pytorch_model.safetensors");
        let p = |suffix: &str| format!("{prefix}.{suffix}");
        Ok(Self {
            norm1: Norm::load(&path, &p("norm1"), context)?,
            self_q: Linear::load(&path, &p("attn1.to_q"), false, context)?,
            self_k: Linear::load(&path, &p("attn1.to_k"), false, context)?,
            self_v: Linear::load(&path, &p("attn1.to_v"), false, context)?,
            self_out: Linear::load(&path, &p("attn1.to_out.0"), true, context)?,
            norm2: Norm::load(&path, &p("norm2"), context)?,
            cross_q: Linear::load(&path, &p("attn2.to_q"), false, context)?,
            cross_k: Linear::load(&path, &p("attn2.to_k"), false, context)?,
            cross_v: Linear::load(&path, &p("attn2.to_v"), false, context)?,
            cross_out: Linear::load(&path, &p("attn2.to_out.0"), true, context)?,
            norm3: Norm::load(&path, &p("norm3"), context)?,
            ff_in: Linear::load(&path, &p("ff.net.0.proj"), true, context)?,
            ff_out: Linear::load(&path, &p("ff.net.2"), true, context)?,
        })
    }

    /// Runs the block on latent tokens and CLIP conditioning tokens.
    pub fn forward(&self, input: &CudaTensor, conditioning: &CudaTensor) -> Result<CudaTensor, String> {
        let normalized = self.norm1.forward(input)?;
        let self_attention = multi_head_attention(
            self.self_q.forward(&normalized)?,
            self.self_k.forward(&normalized)?,
            self.self_v.forward(&normalized)?,
            8,
        )?;
        let hidden = input.add(&self.self_out.forward(&self_attention)?).map_err(|e| format!("self residual: {e:?}"))?;
        let normalized = self.norm2.forward(&hidden)?;
        let cross_attention = multi_head_attention(
            self.cross_q.forward(&normalized)?,
            self.cross_k.forward(conditioning)?,
            self.cross_v.forward(conditioning)?,
            8,
        )?;
        let hidden = hidden.add(&self.cross_out.forward(&cross_attention)?).map_err(|e| format!("cross residual: {e:?}"))?;
        let normalized = self.norm3.forward(&hidden)?;
        let expanded = self.ff_in.forward(&normalized)?;
        let half = expanded.shape()[1] / 2;
        let gated = expanded
            .slice_columns(0, half)
            .map_err(|e| format!("GEGLU value: {e:?}"))?
            .gelu()
            .map_err(|e| format!("GEGLU gelu: {e:?}"))?
            .mul(&expanded.slice_columns(half, half).map_err(|e| format!("GEGLU gate: {e:?}"))?)
            .map_err(|e| format!("GEGLU multiply: {e:?}"))?;
        hidden.add(&self.ff_out.forward(&gated)?).map_err(|e| format!("FFN residual: {e:?}"))
    }
}

fn transpose(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            output[column * rows + row] = values[row * columns + column];
        }
    }
    output
}

fn multi_head_attention(query: CudaTensor, key: CudaTensor, value: CudaTensor, heads: usize) -> Result<CudaTensor, String> {
    let width = *query.shape().get(1).ok_or("attention query must be rank 2")?;
    if heads == 0 || width % heads != 0 || key.shape().get(1) != Some(&width) || value.shape().get(1) != Some(&width) {
        return Err("incompatible multi-head attention shapes".into());
    }
    let head_width = width / heads;
    let mut outputs = Vec::with_capacity(heads);
    for head in 0..heads {
        let offset = head * head_width;
        let q = query.slice_columns(offset, head_width).map_err(|e| format!("attention Q: {e:?}"))?;
        let k = key.slice_columns(offset, head_width).map_err(|e| format!("attention K: {e:?}"))?;
        let v = value.slice_columns(offset, head_width).map_err(|e| format!("attention V: {e:?}"))?;
        outputs.push(q.attention(&k, &v).map_err(|e| format!("attention head: {e:?}"))?);
    }
    CudaTensor::concat_columns(&outputs.iter().collect::<Vec<_>>()).map_err(|e| format!("attention concat: {e:?}"))
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::open_titan_cuda;

    #[test]
    fn executes_real_sd15_mid_block_cross_attention_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let block = TitanBasicTransformerBlock::from_model(&model_dir, "mid_block.attentions.0.transformer_blocks.0", &context)
            .expect("Diffusers transformer weights");
        let latent = CudaTensor::from_slice(context.clone(), vec![16, 1280], &vec![0.0; 16 * 1280]).expect("latent tokens");
        let conditioning = CudaTensor::from_slice(context, vec![77, 768], &vec![0.0; 77 * 768]).expect("CLIP conditioning");
        let output = block.forward(&latent, &conditioning).expect("Titan cross attention");
        assert_eq!(output.shape(), &[16, 1280]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }
}
