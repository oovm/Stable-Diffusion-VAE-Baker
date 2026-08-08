//! Native SD 1.5 VAE decoder entry point on Titan tensors.

use crate::{F32Weight, load_f32_weight};
use std::path::Path;
use titan_hal::CudaContext;
use titan_tensor::{Conv2dOptions, CudaTensor};

fn upload(weight: F32Weight, context: &CudaContext) -> Result<CudaTensor, String> {
    CudaTensor::from_slice(context.clone(), weight.shape, &weight.values).map_err(|e| format!("upload {}: {e:?}", weight.name))
}

fn load(path: &Path, name: &str, context: &CudaContext) -> Result<CudaTensor, String> {
    upload(load_f32_weight(path, name).map_err(|e| e.to_string())?, context)
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
    fn forward(&self, input: &CudaTensor, padding: [usize; 2]) -> Result<CudaTensor, String> {
        input
            .conv2d_nchw(&self.weight, Some(&self.bias), Conv2dOptions { padding, ..Default::default() })
            .map_err(|e| format!("VAE convolution: {e:?}"))
    }
}

/// First executable portion of the SD 1.5 VAE decoder.
pub struct TitanVaeDecoderStem {
    post_quant_conv: Conv,
    decoder_conv_in: Conv,
}

/// VAE residual block shared by decoder mid and up blocks.
pub struct TitanVaeResnetBlock {
    norm1_weight: CudaTensor,
    norm1_bias: CudaTensor,
    conv1: Conv,
    norm2_weight: CudaTensor,
    norm2_bias: CudaTensor,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// One VAE decoder up block with three residual stages and optional upsample.
pub struct TitanVaeUpBlock {
    resnets: Vec<TitanVaeResnetBlock>,
    upsample: Option<Conv>,
}

impl TitanVaeUpBlock {
    /// Loads a decoder up block from the real Diffusers VAE.
    pub fn from_model(
        model_dir: &Path,
        block_index: usize,
        input_channels: usize,
        output_channels: usize,
        context: &CudaContext,
    ) -> Result<Self, String> {
        let path = model_dir.join("vae/diffusion_pytorch_model.safetensors");
        let prefix = format!("decoder.up_blocks.{block_index}");
        let mut resnets = Vec::with_capacity(3);
        for index in 0..3 {
            let input = if index == 0 { input_channels } else { output_channels };
            resnets.push(TitanVaeResnetBlock::from_model(
                model_dir,
                &format!("{prefix}.resnets.{index}"),
                input,
                output_channels,
                context,
            )?);
        }
        let upsample =
            if block_index < 3 { Some(Conv::load(&path, &format!("{prefix}.upsamplers.0.conv"), context)?) } else { None };
        Ok(Self { resnets, upsample })
    }

    /// Runs the VAE block and doubles spatial dimensions when configured.
    pub fn forward(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        let mut hidden = CudaTensor::from_slice(
            input.context(),
            input.shape().to_vec(),
            &input.to_vec().map_err(|e| format!("VAE up input: {e:?}"))?,
        )
        .map_err(|e| format!("VAE up input upload: {e:?}"))?;
        for resnet in &self.resnets {
            hidden = resnet.forward(&hidden)?;
        }
        match &self.upsample {
            Some(upsample) => {
                let [_, _, height, width] = hidden.shape()
                else {
                    return Err("VAE up rank".into());
                };
                let resized =
                    hidden.resize_nearest2d_nchw(*height * 2, *width * 2).map_err(|e| format!("VAE up resize: {e:?}"))?;
                upsample.forward(&resized, [1, 1])
            }
            None => Ok(hidden),
        }
    }
}

impl TitanVaeResnetBlock {
    /// Loads one Diffusers VAE block such as `decoder.mid_block.resnets.0`.
    pub fn from_model(
        model_dir: &Path,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        context: &CudaContext,
    ) -> Result<Self, String> {
        let path = model_dir.join("vae/diffusion_pytorch_model.safetensors");
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
            shortcut,
        })
    }

    /// Runs the VAE block on one NCHW latent batch.
    pub fn forward(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        let residual = match &self.shortcut {
            Some(shortcut) => shortcut.forward(input, [0, 0])?,
            None => CudaTensor::from_slice(
                input.context(),
                input.shape().to_vec(),
                &input.to_vec().map_err(|e| format!("VAE residual: {e:?}"))?,
            )
            .map_err(|e| format!("VAE residual upload: {e:?}"))?,
        };
        let hidden = input
            .group_norm_nchw(32, &self.norm1_weight, &self.norm1_bias, 1e-6)
            .map_err(|e| format!("VAE norm1: {e:?}"))?
            .silu()
            .map_err(|e| format!("VAE silu1: {e:?}"))?;
        let hidden = self.conv1.forward(&hidden, [1, 1])?;
        let hidden = hidden
            .group_norm_nchw(32, &self.norm2_weight, &self.norm2_bias, 1e-6)
            .map_err(|e| format!("VAE norm2: {e:?}"))?
            .silu()
            .map_err(|e| format!("VAE silu2: {e:?}"))?;
        residual.add(&self.conv2.forward(&hidden, [1, 1])?).map_err(|e| format!("VAE residual add: {e:?}"))
    }
}

impl TitanVaeDecoderStem {
    /// Loads the real Diffusers VAE entry weights.
    pub fn from_model_dir(model_dir: &Path, context: &CudaContext) -> Result<Self, String> {
        let path = model_dir.join("vae/diffusion_pytorch_model.safetensors");
        Ok(Self {
            post_quant_conv: Conv::load(&path, "post_quant_conv", context)?,
            decoder_conv_in: Conv::load(&path, "decoder.conv_in", context)?,
        })
    }

    /// Executes the post-quantization projection and decoder input convolution.
    pub fn forward(&self, latent: &CudaTensor) -> Result<CudaTensor, String> {
        let projected = self.post_quant_conv.forward(latent, [0, 0])?;
        self.decoder_conv_in.forward(&projected, [1, 1])
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::open_titan_cuda;

    #[test]
    fn executes_real_sd15_vae_decoder_stem_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let stem = TitanVaeDecoderStem::from_model_dir(&model_dir, &context).expect("VAE stem weights");
        let latent = CudaTensor::from_slice(context, vec![1, 4, 8, 8], &vec![0.0; 4 * 8 * 8]).expect("latent");
        let output = stem.forward(&latent).expect("VAE stem forward");
        assert_eq!(output.shape(), &[1, 512, 8, 8]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn executes_real_sd15_vae_mid_resnet_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("primary context");
        let block = TitanVaeResnetBlock::from_model(&model_dir, "decoder.mid_block.resnets.0", 512, 512, &context)
            .expect("VAE ResNet weights");
        let input = CudaTensor::from_slice(context, vec![1, 512, 4, 4], &vec![0.0; 512 * 4 * 4]).expect("input");
        let output = block.forward(&input).expect("VAE ResNet forward");
        assert_eq!(output.shape(), &[1, 512, 4, 4]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }

    #[test]
    fn executes_real_sd15_vae_up_block_on_gpu() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/sd15");
        let context = open_titan_cuda(0).expect("NVIDIA driver").primary_context().expect("CUDA context");
        let block = TitanVaeUpBlock::from_model(&model_dir, 2, 512, 256, &context).expect("VAE up block weights");
        let input = CudaTensor::from_slice(context, vec![1, 512, 2, 2], &vec![0.0; 512 * 2 * 2]).expect("input");
        let output = block.forward(&input).expect("VAE up block forward");
        assert_eq!(output.shape(), &[1, 256, 4, 4]);
        assert!(output.to_vec().expect("download").iter().all(|value| value.is_finite()));
    }
}
