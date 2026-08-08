//! Device-resident DDIM scheduling for native Titan SD 1.5 inference.

use titan_tensor::CudaTensor;

/// Deterministic (`eta = 0`) DDIM schedule using the SD 1.x linear beta range.
#[derive(Clone, Debug)]
pub struct TitanDdimScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f32>,
}

impl TitanDdimScheduler {
    pub fn new(steps: usize, train_steps: usize) -> Result<Self, String> {
        if steps == 0 || train_steps < 2 || steps > train_steps {
            return Err("invalid DDIM step count".into());
        }
        let mut alpha = 1.0;
        let mut alphas_cumprod = Vec::with_capacity(train_steps);
        for index in 0..train_steps {
            let beta = 0.00085 + (0.012 - 0.00085) * index as f32 / (train_steps - 1) as f32;
            alpha *= 1.0 - beta;
            alphas_cumprod.push(alpha);
        }
        let timesteps = (0..steps).map(|index| train_steps - 1 - index * train_steps / steps).collect();
        Ok(Self { timesteps, alphas_cumprod })
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Applies one DDIM update without downloading the latent from CUDA.
    pub fn step(&self, sample: &CudaTensor, noise_prediction: &CudaTensor, step: usize) -> Result<CudaTensor, String> {
        if sample.shape() != noise_prediction.shape() || step >= self.timesteps.len() {
            return Err("DDIM sample/noise shape mismatch".into());
        }
        let timestep = self.timesteps[step];
        let previous = if step + 1 < self.timesteps.len() { self.timesteps[step + 1] } else { 0 };
        let alpha = self.alphas_cumprod[timestep];
        let previous_alpha = self.alphas_cumprod[previous];
        let pred_x0 = sample
            .scale(1.0 / alpha.sqrt())
            .map_err(|e| format!("DDIM scale: {e:?}"))?
            .add(&noise_prediction.scale(-((1.0 - alpha).sqrt() / alpha.sqrt())).map_err(|e| format!("DDIM scale: {e:?}"))?)
            .map_err(|e| format!("DDIM prediction: {e:?}"))?;
        pred_x0
            .scale(previous_alpha.sqrt())
            .map_err(|e| format!("DDIM scale: {e:?}"))?
            .add(&noise_prediction.scale((1.0 - previous_alpha).sqrt()).map_err(|e| format!("DDIM scale: {e:?}"))?)
            .map_err(|error| format!("DDIM device step: {error:?}"))
    }
}
