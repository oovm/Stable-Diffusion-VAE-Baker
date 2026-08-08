//! Safe model inspection and format detection.
use diffusion_types::{DiffusionError, ModelFamily, Result};
use safetensors::SafeTensors;
use std::{collections::BTreeSet, collections::BTreeMap, fs, path::Path};
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
    } else if keys.iter().any(|k| k.contains("model.diffusion_model.input_blocks.0.0.weight")) && keys.iter().any(|k| k.contains("input_blocks.0.0.weight")) {
        ModelFamily::Sd15
    } else {
        ModelFamily::Sd15
    };
    let derivative = if keys.iter().any(|k| k.to_ascii_lowercase().contains("pony")) { Some("pony".into()) } else { None };
    let mut components = BTreeMap::new();
    for key in &keys {
        let component = if key.starts_with("model.diffusion_model.") || key.starts_with("unet.") { "unet" }
            else if key.starts_with("first_stage_model.") || key.starts_with("vae.") { "vae" }
            else if key.starts_with("cond_stage_model.") || key.starts_with("conditioner.") { "text_encoder" }
            else { "other" };
        *components.entry(component.into()).or_insert(0) += 1;
    }
    let metadata = BTreeMap::new();
    Ok(ModelInfo { family, derivative, tensors: keys.len(), keys, components, metadata })
}
