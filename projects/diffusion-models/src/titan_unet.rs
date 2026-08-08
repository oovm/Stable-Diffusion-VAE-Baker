//! Native Stable Diffusion 1.5 UNet building blocks on Titan tensors.
//!
//! This module keeps Diffusers names and layouts explicit. It starts with the
//! ResNet block shared by SD down, mid, and up blocks; attention and block
//! graph wiring are layered on this contract.

use crate::{F32Weight, load_f32_weight};
use std::path::Path;
use titan_hal::CudaContext;
use titan_tensor::{Conv2dOptions, CudaTensor};

fn upload(weight: F32Weight, context: &CudaContext) -> Result<CudaTensor, String> {
    CudaTensor::from_slice(context.clone(), weight.shape, &weight.values)
        .map_err(|error| format!("upload {}: {error:?}", weight.name))
}

fn load(path: &Path, name: &str, context: &CudaContext) -> Result<CudaTensor, String> {
    upload(load_f32_weight(path, name).map_err(|error| error.to_string())?, context)
}

struct Conv {
    weight: CudaTensor,
    bias: CudaTensor,
}

impl Conv {
    fn load(path: &Path, prefix: &str, context: &CudaContext) -> Result<Self, String> {
        Ok(Self {
            weight: load(path, &format!("{prefix}.weight"), context)?,
            bias: load(path, &format!("{prefix}.bias"), context)?,
        })
    }

    fn forward(&self, input: &CudaTensor, stride: [usize; 2]) -> Result<CudaTensor, String> {
        input
            .conv2d_nchw(&self.weight, Some(&self.bias), Conv2dOptions { stride, padding: [1, 1], ..Default::default() })
            .map_err(|error| format!("convolution: {error:?}"))
    }
}

struct TimeProjection {
    weight: CudaTensor,
    bias: CudaTensor,
}

impl TimeProjection {
    fn load(path: &Path, prefix: &str, context: &CudaContext) -> Result<Self, String> {
        let weight = load(path, &format!("{prefix}.weight"), context)?
            .transpose()
            .map_err(|error| format!("time projection transpose: {error:?}"))?;
        Ok(Self { weight, bias: load(path, &format!("{prefix}.bias"), context)? })
    }

    fn forward(&self, time: &CudaTensor) -> Result<CudaTensor, String> {
        time.matmul(&self.weight)
            .and_then(|value| value.add_bias(&self.bias))
            .map_err(|error| format!("time projection: {error:?}"))
    }
}

/// A Diffusers SD 1.5 ResNet block with time embedding injection.
pub struct TitanResnetBlock {
    norm1_weight: CudaTensor,
    norm1_bias: CudaTensor,
    conv1: Conv,
    norm2_weight: CudaTensor,
    norm2_bias: CudaTensor,
    conv2: Conv,
    time_projection: TimeProjection,
    shortcut: Option<Conv>,
}

impl TitanResnetBlock {
    /// Loads one `down_blocks.X.resnets.Y` or `mid_block.resnets.Y` block.
    pub fn from_model(
        model_dir: &Path,
        prefix: &str,
        context: &CudaContext,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self, String> {
        let path = model_dir.join("unet/diffusion_pytorch_model.safetensors");
        let shortcut = if input_channels != output_channels {
            Some(Conv::load(&path, &format!("{prefix}.conv_shortcut"), context)?)
        }
        else {
            None
        };
        Ok(Self {
            norm1_weight: load(&path, &format!("{prefix}.norm1.weight"), context)?,
            norm1_bias: load(&path, &format!("{prefix}.norm1.bias"), context)?,
            conv1: Conv::load(&path, &format!("{prefix}.conv1"), context)?,
            norm2_weight: load(&path, &format!("{prefix}.norm2.weight"), context)?,
            norm2_bias: load(&path, &format!("{prefix}.norm2.bias"), context)?,
            conv2: Conv::load(&path, &format!("{prefix}.conv2"), context)?,
            time_projection: TimeProjection::load(&path, &format!("{prefix}.time_emb_proj"), context)?,
            shortcut,
        })
    }

    /// Executes the block on `[batch, channels, height, width]` tensors.
    pub fn forward(&self, input: &CudaTensor, time_embedding: &CudaTensor) -> Result<CudaTensor, String> {
        let residual = match &self.shortcut {
            Some(shortcut) => shortcut.forward(input, [1, 1])?,
            None => CudaTensor::from_slice(
                input.context(),
                input.shape().to_vec(),
                &input.to_vec().map_err(|error| format!("residual readback: {error:?}"))?,
            )
            .map_err(|error| format!("residual: {error:?}"))?,
        };
        let hidden = input
            .group_norm_nchw(32, &self.norm1_weight, &self.norm1_bias, 1e-5)
            .map_err(|error| format!("group norm 1: {error:?}"))?
            .silu()
            .map_err(|error| format!("silu 1: {error:?}"))?;
        let hidden = self.conv1.forward(&hidden, [1, 1])?;
        let time = self.time_projection.forward(time_embedding)?;
        let time_values = time.to_vec().map_err(|error| format!("time readback: {error:?}"))?;
        let time = CudaTensor::from_slice(input.context(), vec![time_values.len()], &time_values)
            .map_err(|error| format!("time broadcast: {error:?}"))?;
        let hidden = hidden.add_channels_nchw(&time).map_err(|error| format!("time injection: {error:?}"))?;
        let hidden = hidden
            .group_norm_nchw(32, &self.norm2_weight, &self.norm2_bias, 1e-5)
            .map_err(|error| format!("group norm 2: {error:?}"))?
            .silu()
            .map_err(|error| format!("silu 2: {error:?}"))?;
        let hidden = self.conv2.forward(&hidden, [1, 1])?;
        residual.add(&hidden).map_err(|error| format!("resnet residual: {error:?}"))
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::open_titan_cuda;

    #[test]
    fn loads_real_sd15_first_unet_resnet_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let block = TitanResnetBlock::from_model(&model_dir, "down_blocks.0.resnets.0", &context, 320, 320)
            .expect("first SD15 ResNet weights");
        assert_eq!(block.norm1_weight.shape(), &[320]);
        assert_eq!(block.conv1.weight.shape(), &[320, 320, 3, 3]);
        assert_eq!(block.time_projection.weight.shape(), &[1280, 320]);
    }
}
