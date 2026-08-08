//! Native Stable Diffusion 1.5 UNet building blocks on Titan tensors.
//!
//! This module keeps Diffusers names and layouts explicit. It starts with the
//! ResNet block shared by SD down, mid, and up blocks; attention and block
//! graph wiring are layered on this contract.

use crate::{F32Weight, load_f32_weight, titan_attention::TitanSpatialTransformer};
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

    fn forward(&self, input: &CudaTensor, stride: [usize; 2], padding: [usize; 2]) -> Result<CudaTensor, String> {
        input
            .conv2d_nchw(&self.weight, Some(&self.bias), Conv2dOptions { stride, padding, ..Default::default() })
            .map_err(|error| format!("convolution: {error:?}"))
    }
}

struct TimeProjection {
    weight: CudaTensor,
    bias: CudaTensor,
}

/// SD 1.5 sinusoidal timestep embedding followed by the two learned layers.
pub struct TitanTimeEmbedding {
    first: TimeProjection,
    second: TimeProjection,
    context: CudaContext,
}

impl TitanTimeEmbedding {
    /// Loads `time_embedding.linear_1` and `time_embedding.linear_2`.
    pub fn from_model(model_dir: &Path, context: &CudaContext) -> Result<Self, String> {
        let path = model_dir.join("unet/diffusion_pytorch_model.safetensors");
        Ok(Self {
            first: TimeProjection::load(&path, "time_embedding.linear_1", context)?,
            second: TimeProjection::load(&path, "time_embedding.linear_2", context)?,
            context: context.clone(),
        })
    }

    /// Encodes one denoising timestep into the ResNet conditioning vector.
    pub fn forward(&self, timestep: f32) -> Result<CudaTensor, String> {
        let frequencies = 160usize;
        let mut values = Vec::with_capacity(frequencies * 2);
        for index in 0..frequencies {
            let frequency = (-((10_000.0_f32).ln()) * index as f32 / (frequencies - 1) as f32).exp();
            values.push((timestep * frequency).sin());
        }
        for index in 0..frequencies {
            let frequency = (-((10_000.0_f32).ln()) * index as f32 / (frequencies - 1) as f32).exp();
            values.push((timestep * frequency).cos());
        }
        let input = CudaTensor::from_slice(self.context.clone(), vec![1, frequencies * 2], &values)
            .map_err(|error| format!("timestep upload: {error:?}"))?;
        let hidden = self.first.forward(&input)?.silu().map_err(|error| format!("timestep SiLU: {error:?}"))?;
        self.second.forward(&hidden)
    }
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
            Some(shortcut) => shortcut.forward(input, [1, 1], [0, 0])?,
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
        let hidden = self.conv1.forward(&hidden, [1, 1], [1, 1])?;
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
        let hidden = self.conv2.forward(&hidden, [1, 1], [1, 1])?;
        residual.add(&hidden).map_err(|error| format!("resnet residual: {error:?}"))
    }
}

/// The initial executable SD 1.5 UNet path up to the first down-block ResNet.
pub struct TitanUnetStem {
    conv_in: Conv,
    time_embedding: TitanTimeEmbedding,
    first_resnet: TitanResnetBlock,
}

/// A Diffusers UNet down block with two time-conditioned ResNets and a stride-2 convolution.
pub struct TitanDownBlock {
    resnets: Vec<TitanResnetBlock>,
    downsample: Conv,
}

impl TitanDownBlock {
    /// Loads a two-ResNet down block, including `downsamplers.0.conv`.
    pub fn from_model(
        model_dir: &Path,
        block_index: usize,
        input_channels: usize,
        output_channels: usize,
        context: &CudaContext,
    ) -> Result<Self, String> {
        let path = model_dir.join("unet/diffusion_pytorch_model.safetensors");
        let prefix = format!("down_blocks.{block_index}");
        Ok(Self {
            resnets: vec![
                TitanResnetBlock::from_model(
                    model_dir,
                    &format!("{prefix}.resnets.0"),
                    context,
                    input_channels,
                    output_channels,
                )?,
                TitanResnetBlock::from_model(
                    model_dir,
                    &format!("{prefix}.resnets.1"),
                    context,
                    output_channels,
                    output_channels,
                )?,
            ],
            downsample: Conv::load(&path, &format!("{prefix}.downsamplers.0.conv"), context)?,
        })
    }

    /// Runs both residual blocks and halves the spatial dimensions.
    pub fn forward(&self, input: &CudaTensor, time: &CudaTensor) -> Result<CudaTensor, String> {
        let mut hidden = CudaTensor::from_slice(
            input.context(),
            input.shape().to_vec(),
            &input.to_vec().map_err(|e| format!("down block input: {e:?}"))?,
        )
        .map_err(|e| format!("down block upload: {e:?}"))?;
        for block in &self.resnets {
            hidden = block.forward(&hidden, time)?;
        }
        self.downsample.forward(&hidden, [2, 2], [1, 1])
    }
}

/// A cross-attention down block used by SD 1.5 blocks 1 and 2.
pub struct TitanCrossAttnDownBlock {
    first_resnet: TitanResnetBlock,
    first_attention: TitanSpatialTransformer,
    second_resnet: TitanResnetBlock,
    second_attention: TitanSpatialTransformer,
    downsample: Conv,
}

/// SD 1.5's bottleneck: ResNet, spatial transformer, then ResNet.
pub struct TitanMidBlock {
    first_resnet: TitanResnetBlock,
    attention: TitanSpatialTransformer,
    second_resnet: TitanResnetBlock,
}

/// The first SD 1.5 up block with three skip-connected ResNet stages.
pub struct TitanUpBlock {
    resnets: Vec<TitanResnetBlock>,
    upsample: Conv,
}

impl TitanUpBlock {
    /// Loads `up_blocks.0`, whose three ResNets consume 1280-channel skips.
    pub fn from_model(model_dir: &Path, context: &CudaContext) -> Result<Self, String> {
        let path = model_dir.join("unet/diffusion_pytorch_model.safetensors");
        let prefix = "up_blocks.0";
        let mut resnets = Vec::with_capacity(3);
        for index in 0..3 {
            resnets.push(TitanResnetBlock::from_model(model_dir, &format!("{prefix}.resnets.{index}"), context, 2560, 1280)?);
        }
        Ok(Self { resnets, upsample: Conv::load(&path, &format!("{prefix}.upsamplers.0.conv"), context)? })
    }

    /// Runs three skip-connected stages then doubles the spatial dimensions.
    pub fn forward(
        &self,
        input: &CudaTensor,
        skips: [&CudaTensor; 3],
        time: &CudaTensor,
        _conditioning: &CudaTensor,
    ) -> Result<CudaTensor, String> {
        let mut hidden = CudaTensor::from_slice(
            input.context(),
            input.shape().to_vec(),
            &input.to_vec().map_err(|e| format!("up input: {e:?}"))?,
        )
        .map_err(|e| format!("up input upload: {e:?}"))?;
        for (resnet, skip) in self.resnets.iter().zip(skips) {
            let merged = CudaTensor::concat_channels_nchw(&[&hidden, skip]).map_err(|e| format!("up skip concat: {e:?}"))?;
            hidden = resnet.forward(&merged, time)?;
        }
        let [_, _channels, height, width] = hidden.shape()
        else {
            return Err("up block returned invalid rank".into());
        };
        let upsampled = hidden.resize_nearest2d_nchw(*height * 2, *width * 2).map_err(|e| format!("up resize: {e:?}"))?;
        self.upsample.forward(&upsampled, [1, 1], [1, 1])
    }
}

impl TitanMidBlock {
    /// Loads `mid_block.resnets.0`, `mid_block.attentions.0`, and `mid_block.resnets.1`.
    pub fn from_model(model_dir: &Path, context: &CudaContext) -> Result<Self, String> {
        Ok(Self {
            first_resnet: TitanResnetBlock::from_model(model_dir, "mid_block.resnets.0", context, 1280, 1280)?,
            attention: TitanSpatialTransformer::from_model(model_dir, "mid_block.attentions.0", 1280, context)?,
            second_resnet: TitanResnetBlock::from_model(model_dir, "mid_block.resnets.1", context, 1280, 1280)?,
        })
    }

    /// Executes the complete conditioned UNet bottleneck.
    pub fn forward(&self, input: &CudaTensor, time: &CudaTensor, conditioning: &CudaTensor) -> Result<CudaTensor, String> {
        let hidden = self.first_resnet.forward(input, time)?;
        let hidden = self.attention.forward(&hidden, conditioning)?;
        self.second_resnet.forward(&hidden, time)
    }
}

impl TitanCrossAttnDownBlock {
    /// Loads the two ResNet/Transformer pairs and the stride-2 downsampler.
    pub fn from_model(
        model_dir: &Path,
        block_index: usize,
        input_channels: usize,
        output_channels: usize,
        context: &CudaContext,
    ) -> Result<Self, String> {
        let path = model_dir.join("unet/diffusion_pytorch_model.safetensors");
        let prefix = format!("down_blocks.{block_index}");
        Ok(Self {
            first_resnet: TitanResnetBlock::from_model(
                model_dir,
                &format!("{prefix}.resnets.0"),
                context,
                input_channels,
                output_channels,
            )?,
            first_attention: TitanSpatialTransformer::from_model(
                model_dir,
                &format!("{prefix}.attentions.0"),
                output_channels,
                context,
            )?,
            second_resnet: TitanResnetBlock::from_model(
                model_dir,
                &format!("{prefix}.resnets.1"),
                context,
                output_channels,
                output_channels,
            )?,
            second_attention: TitanSpatialTransformer::from_model(
                model_dir,
                &format!("{prefix}.attentions.1"),
                output_channels,
                context,
            )?,
            downsample: Conv::load(&path, &format!("{prefix}.downsamplers.0.conv"), context)?,
        })
    }

    /// Executes the complete cross-attention down block.
    pub fn forward(&self, input: &CudaTensor, time: &CudaTensor, conditioning: &CudaTensor) -> Result<CudaTensor, String> {
        let hidden = self.first_resnet.forward(input, time)?;
        let hidden = self.first_attention.forward(&hidden, conditioning)?;
        let hidden = self.second_resnet.forward(&hidden, time)?;
        let hidden = self.second_attention.forward(&hidden, conditioning)?;
        self.downsample.forward(&hidden, [2, 2], [1, 1])
    }
}

impl TitanUnetStem {
    /// Loads `conv_in`, learned timestep layers, and `down_blocks.0.resnets.0`.
    pub fn from_model(model_dir: &Path, context: &CudaContext) -> Result<Self, String> {
        let path = model_dir.join("unet/diffusion_pytorch_model.safetensors");
        Ok(Self {
            conv_in: Conv::load(&path, "conv_in", context)?,
            time_embedding: TitanTimeEmbedding::from_model(model_dir, context)?,
            first_resnet: TitanResnetBlock::from_model(model_dir, "down_blocks.0.resnets.0", context, 320, 320)?,
        })
    }

    /// Runs `[1,4,H,W]` latent input through the first UNet residual block.
    pub fn forward(&self, latent: &CudaTensor, timestep: f32) -> Result<CudaTensor, String> {
        let hidden = self.conv_in.forward(latent, [1, 1], [1, 1])?;
        let time = self.time_embedding.forward(timestep)?;
        self.first_resnet.forward(&hidden, &time)
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

    #[test]
    fn executes_real_sd15_first_resnet_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let block = TitanResnetBlock::from_model(&model_dir, "down_blocks.0.resnets.0", &context, 320, 320)
            .expect("first SD15 ResNet weights");
        let input = CudaTensor::from_slice(context.clone(), vec![1, 320, 8, 8], &vec![0.0; 320 * 8 * 8]).expect("latent input");
        let time = CudaTensor::from_slice(context, vec![1, 1280], &vec![0.0; 1280]).expect("time embedding");
        let output = block.forward(&input, &time).expect("Titan ResNet forward");
        assert_eq!(output.shape(), &[1, 320, 8, 8]);
        assert!(output.to_vec().expect("output download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn executes_real_sd15_timestep_embedding_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let embedding = TitanTimeEmbedding::from_model(&model_dir, &context).expect("time embedding weights");
        let output = embedding.forward(999.0).expect("timestep forward");
        assert_eq!(output.shape(), &[1, 1280]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn connects_real_sd15_timestep_to_first_resnet_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let time =
            TitanTimeEmbedding::from_model(&model_dir, &context).expect("time embedding").forward(999.0).expect("time forward");
        let block = TitanResnetBlock::from_model(&model_dir, "down_blocks.0.resnets.0", &context, 320, 320).expect("resnet");
        let input = CudaTensor::from_slice(context, vec![1, 320, 4, 4], &vec![0.0; 320 * 4 * 4]).expect("input");
        let output = block.forward(&input, &time).expect("conditioned ResNet");
        assert_eq!(output.shape(), &[1, 320, 4, 4]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn executes_real_sd15_unet_stem_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let stem = TitanUnetStem::from_model(&model_dir, &context).expect("UNet stem weights");
        let latent = CudaTensor::from_slice(context, vec![1, 4, 4, 4], &vec![0.0; 4 * 4 * 4]).expect("latent");
        let output = stem.forward(&latent, 999.0).expect("UNet stem forward");
        assert_eq!(output.shape(), &[1, 320, 4, 4]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn executes_real_sd15_first_down_block_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let time =
            TitanTimeEmbedding::from_model(&model_dir, &context).expect("time embedding").forward(999.0).expect("time forward");
        let block = TitanDownBlock::from_model(&model_dir, 0, 320, 320, &context).expect("down block weights");
        let input = CudaTensor::from_slice(context, vec![1, 320, 4, 4], &vec![0.0; 320 * 4 * 4]).expect("input");
        let output = block.forward(&input, &time).expect("down block forward");
        assert_eq!(output.shape(), &[1, 320, 2, 2]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn executes_real_sd15_cross_attention_down_block_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let time =
            TitanTimeEmbedding::from_model(&model_dir, &context).expect("time embedding").forward(999.0).expect("time forward");
        let block = TitanCrossAttnDownBlock::from_model(&model_dir, 1, 320, 640, &context).expect("cross down block weights");
        let input = CudaTensor::from_slice(context.clone(), vec![1, 320, 4, 4], &vec![0.0; 320 * 4 * 4]).expect("input");
        let conditioning = CudaTensor::from_slice(context, vec![77, 768], &vec![0.0; 77 * 768]).expect("conditioning");
        let output = block.forward(&input, &time, &conditioning).expect("cross down block forward");
        assert_eq!(output.shape(), &[1, 640, 2, 2]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn executes_real_sd15_mid_block_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let time =
            TitanTimeEmbedding::from_model(&model_dir, &context).expect("time embedding").forward(999.0).expect("time forward");
        let block = TitanMidBlock::from_model(&model_dir, &context).expect("mid block weights");
        let input = CudaTensor::from_slice(context.clone(), vec![1, 1280, 2, 2], &vec![0.0; 1280 * 2 * 2]).expect("input");
        let conditioning = CudaTensor::from_slice(context, vec![77, 768], &vec![0.0; 77 * 768]).expect("conditioning");
        let output = block.forward(&input, &time, &conditioning).expect("mid block forward");
        assert_eq!(output.shape(), &[1, 1280, 2, 2]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn executes_real_sd15_first_up_block_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let time =
            TitanTimeEmbedding::from_model(&model_dir, &context).expect("time embedding").forward(999.0).expect("time forward");
        let block = TitanUpBlock::from_model(&model_dir, &context).expect("up block weights");
        let input = CudaTensor::from_slice(context.clone(), vec![1, 1280, 2, 2], &vec![0.0; 1280 * 2 * 2]).expect("input");
        let skip1 = CudaTensor::from_slice(context.clone(), vec![1, 1280, 2, 2], &vec![0.0; 1280 * 2 * 2]).expect("skip1");
        let skip2 = CudaTensor::from_slice(context.clone(), vec![1, 1280, 2, 2], &vec![0.0; 1280 * 2 * 2]).expect("skip2");
        let skip3 = CudaTensor::from_slice(context.clone(), vec![1, 1280, 2, 2], &vec![0.0; 1280 * 2 * 2]).expect("skip3");
        let conditioning = CudaTensor::from_slice(context, vec![77, 768], &vec![0.0; 77 * 768]).expect("conditioning");
        let output = block.forward(&input, [&skip1, &skip2, &skip3], &time, &conditioning).expect("up block forward");
        assert_eq!(output.shape(), &[1, 1280, 4, 4]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }
}
