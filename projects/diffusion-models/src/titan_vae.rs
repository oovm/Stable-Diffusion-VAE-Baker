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
}
