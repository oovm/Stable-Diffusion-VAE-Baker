//! Shared public types for the lightweight diffusion workspace.
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum DiffusionError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("cancelled")]
    Cancelled,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, DiffusionError>;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelFamily {
    Sd15,
    Sd21,
    Sdxl,
}
/// Runtime backend selection exposed by the single `sd.exe` binary.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DevicePreference {
    #[default]
    Auto,
    Cpu,
    Cuda,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRef {
    pub path: PathBuf,
    pub family: Option<ModelFamily>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub weight: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingSpec {
    pub path: PathBuf,
    pub token: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditioningInput {
    pub path: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnotatorSpec {
    pub name: String,
    pub weights: Option<PathBuf>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlNetSpec {
    pub model: PathBuf,
    pub image: PathBuf,
    pub weight: f32,
    pub start: f32,
    pub end: f32,
}
#[derive(Clone, Debug)]
pub struct ImageInput(pub DynamicImage);
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
    pub loras: Vec<LoraSpec>,
    pub embeddings: Vec<EmbeddingSpec>,
    pub conditioning: Option<ConditioningInput>,
    pub controlnets: Vec<ControlNetSpec>,
    pub annotator: Option<AnnotatorSpec>,
    pub vae: Option<ModelRef>,
}
impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            width: 512,
            height: 512,
            steps: 20,
            guidance_scale: 7.5,
            seed: None,
            loras: vec![],
            embeddings: vec![],
            conditioning: None,
            controlnets: vec![],
            annotator: None,
            vae: None,
        }
    }
}
impl GenerationRequest {
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width % 8 != 0 || self.height % 8 != 0 {
            return Err(DiffusionError::InvalidRequest("width/height must be non-zero multiples of 8".into()));
        }
        if self.steps == 0 || !self.guidance_scale.is_finite() {
            return Err(DiffusionError::InvalidRequest("steps must be positive and guidance_scale finite".into()));
        }
        for control in &self.controlnets {
            if !control.weight.is_finite()
                || !(0.0..=1.0).contains(&control.start)
                || !(0.0..=1.0).contains(&control.end)
                || control.start > control.end
            {
                return Err(DiffusionError::InvalidRequest("invalid ControlNet weight or step range".into()));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub images: Vec<DynamicImage>,
    pub seed: u64,
}
pub trait ProgressSink: Send + Sync {
    fn step(&self, current: u32, total: u32);
}
#[derive(Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release)
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}
pub trait DiffusionPipeline: Send + Sync {
    fn generate(
        &self,
        request: &GenerationRequest,
        progress: &dyn ProgressSink,
        cancel: &CancellationToken,
    ) -> Result<GenerationResult>;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_control_range() {
        let mut request = GenerationRequest::default();
        request.controlnets.push(ControlNetSpec {
            model: PathBuf::from("control.safetensors"),
            image: PathBuf::from("control.png"),
            weight: 1.0,
            start: 0.8,
            end: 0.2,
        });
        assert!(request.validate().is_err());
    }
    #[test]
    fn accepts_default_request() {
        assert!(GenerationRequest::default().validate().is_ok());
    }
}
