//! Safe model inspection and format detection.
use diffusion_types::{DiffusionError, ModelFamily, Result};
use safetensors::SafeTensors;
use std::{collections::BTreeSet, fs, path::Path};
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub family: ModelFamily,
    pub tensors: usize,
    pub keys: BTreeSet<String>,
}
pub fn inspect(path: &Path) -> Result<ModelInfo> {
    let file = if path.is_dir() { path.join("model.safetensors") } else { path.to_path_buf() };
    let bytes = fs::read(&file).map_err(|e| DiffusionError::Model(format!("{}: {e}", file.display())))?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| DiffusionError::Model(e.to_string()))?;
    let keys: BTreeSet<_> = st.names().into_iter().map(|key| key.to_owned()).collect();
    let family = if keys.iter().any(|k| k.contains("conditioner.embedders.1")) { ModelFamily::Sdxl } else { ModelFamily::Sd15 };
    Ok(ModelInfo { family, tensors: keys.len(), keys })
}
