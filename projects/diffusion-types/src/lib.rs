//! Shared public types for the lightweight diffusion workspace.
use candle_core::{DType, Device};
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
    Sdxl,
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
    pub controlnets: Vec<ControlNetSpec>,
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
            controlnets: vec![],
            vae: None,
        }
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
pub fn select_device(name: Option<&str>) -> Result<(Device, DType)> {
    match name.unwrap_or("cpu") {
        "cpu" => Ok((Device::Cpu, DType::F32)),
        other => Err(DiffusionError::Model(format!("backend `{other}` is not enabled in this build"))),
    }
}
