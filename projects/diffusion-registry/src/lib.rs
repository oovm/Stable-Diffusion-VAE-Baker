//! Well-known local model sources, parallel downloads, and resumable transfers.
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::{
    StatusCode,
    blocking::{Client, Response},
    header::{AUTHORIZATION, CONTENT_RANGE, HeaderValue, RANGE},
};
use serde::{Deserialize, Serialize};
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("unknown model `{0}`")]
    UnknownModel(String),
    #[error("network: {0}")]
    Network(#[from] reqwest::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("metadata: {0}")]
    Metadata(#[from] serde_json::Error),
    #[error("download worker panicked")]
    WorkerPanic,
}
pub type Result<T> = std::result::Result<T, RegistryError>;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFile {
    pub path: String,
    pub url: String,
    pub expected_bytes: Option<u64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSource {
    pub id: String,
    pub display_name: String,
    pub family: String,
    pub revision: String,
    pub files: Vec<SourceFile>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DownloadFile {
    pub path: String,
    pub downloaded_bytes: u64,
    pub expected_bytes: Option<u64>,
    pub complete: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DownloadMetadata {
    pub model: ModelSource,
    pub files: Vec<DownloadFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentIntegrity {
    pub path: String,
    pub exists: bool,
    pub actual_bytes: Option<u64>,
    pub expected_bytes: Option<u64>,
    pub safetensors_valid: Option<bool>,
    pub sha256: Option<String>,
    pub complete: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelIntegrityReport {
    pub model_id: String,
    pub root: PathBuf,
    pub legacy_manifest_migrated: bool,
    pub components: Vec<ComponentIntegrity>,
}

impl ModelIntegrityReport {
    pub fn complete(&self) -> bool { self.components.iter().all(|component| component.complete) }
}

fn source(path: &str, url: &str, expected_bytes: Option<u64>) -> SourceFile {
    SourceFile { path: path.into(), url: url.into(), expected_bytes }
}

fn is_safetensors(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    // Keep the streaming buffer on the heap: Windows main-thread stacks can be smaller than 1 MiB.
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 { break; }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn inspect_component(root: &Path, source: &SourceFile) -> ComponentIntegrity {
    let path = root.join(&source.path);
    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return ComponentIntegrity {
            path: source.path.clone(), exists: false, actual_bytes: None, expected_bytes: source.expected_bytes,
            safetensors_valid: None, sha256: None, complete: false, detail: "missing".into(),
        },
        Err(error) => return ComponentIntegrity {
            path: source.path.clone(), exists: false, actual_bytes: None, expected_bytes: source.expected_bytes,
            safetensors_valid: None, sha256: None, complete: false, detail: error.to_string(),
        },
    };
    let size_matches = source.expected_bytes.is_none_or(|expected| expected == metadata.len());
    let safetensors_valid = if is_safetensors(&path) {
        Some(match fs::read(&path) {
            Ok(bytes) => SafeTensors::deserialize(&bytes).is_ok(),
            Err(_) => false,
        })
    } else { None };
    let parse_matches = safetensors_valid.unwrap_or(true);
    let complete = size_matches && parse_matches;
    let detail = if complete { "trusted metadata and local file agree".into() }
        else if !size_matches { format!("size mismatch: expected {:?}, got {}", source.expected_bytes, metadata.len()) }
        else { "safetensors header or tensor table is invalid".into() };
    ComponentIntegrity {
        path: source.path.clone(), exists: true, actual_bytes: Some(metadata.len()), expected_bytes: source.expected_bytes,
        safetensors_valid, sha256: sha256_file(&path).ok(), complete, detail,
    }
}

pub fn verify(id: &str, root: impl AsRef<Path>) -> Result<ModelIntegrityReport> {
    let model = well_known(id)?;
    let root = root.as_ref().to_path_buf();
    let legacy_manifest = root.join(".sd-download.json");
    let legacy_manifest_migrated = fs::read(&legacy_manifest).ok()
        .and_then(|bytes| serde_json::from_slice::<DownloadMetadata>(&bytes).ok())
        .is_some_and(|metadata| metadata.model.files != model.files);
    let report = ModelIntegrityReport {
        model_id: model.id.clone(),
        root: root.clone(),
        legacy_manifest_migrated,
        components: model.files.iter().map(|source| inspect_component(&root, source)).collect(),
    };
    if report.legacy_manifest_migrated {
        let files = report.components.iter().map(|component| DownloadFile {
            path: component.path.clone(),
            downloaded_bytes: component.actual_bytes.unwrap_or(0),
            expected_bytes: component.expected_bytes,
            complete: component.complete,
        }).collect();
        write_manifest_atomic(&root, &DownloadMetadata { model, files })?;
    }
    Ok(report)
}

fn write_manifest_atomic(root: &Path, metadata: &DownloadMetadata) -> Result<()> {
    let destination = root.join(".sd-download.json");
    let temporary = root.join(".sd-download.json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(metadata)?)?;
    fs::rename(&temporary, destination)?;
    Ok(())
}

pub fn well_known(id: &str) -> Result<ModelSource> {
    match id {
        "sd15" | "stable-diffusion-v1-5" => Ok(ModelSource { id: "sd15".into(), display_name: "Stable Diffusion v1.5".into(), family: "sd15".into(), revision: "main".into(), files: vec![ source("tokenizer/tokenizer.json", "https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/tokenizer.json", Some(2_224_041)), source("text_encoder/model.safetensors", "https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/text_encoder/model.safetensors", Some(492_265_874)), source("unet/diffusion_pytorch_model.safetensors", "https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.safetensors", Some(3_438_167_540)), source("vae/diffusion_pytorch_model.safetensors", "https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.safetensors", Some(334_643_276)) ] }),
        "sd21" | "stable-diffusion-2-1-base" => Ok(ModelSource { id: "sd21".into(), display_name: "Stable Diffusion 2.1 Base".into(), family: "sd21".into(), revision: "main".into(), files: vec![ source("tokenizer/tokenizer.json", "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json", None), source("text_encoder/model.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/text_encoder/model.safetensors", None), source("unet/diffusion_pytorch_model.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/unet/diffusion_pytorch_model.safetensors", None), source("vae/diffusion_pytorch_model.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/vae/diffusion_pytorch_model.safetensors", None) ] }),
        "sdxl" | "sdxl-base-1.0" => Ok(ModelSource { id: "sdxl".into(), display_name: "Stable Diffusion XL Base 1.0".into(), family: "sdxl".into(), revision: "main".into(), files: vec![ source("tokenizer/tokenizer.json", "https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/tokenizer.json", Some(2_224_041)), source("tokenizer_2/tokenizer.json", "https://hf-mirror.com/openai/clip-vit-large-patch14/resolve/main/tokenizer.json", None), source("text_encoder/model.safetensors", "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.safetensors", Some(492_265_168)), source("text_encoder_2/model.safetensors", "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors", Some(2_778_702_264)), source("unet/diffusion_pytorch_model.safetensors", "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors", Some(10_270_077_736)), source("vae/diffusion_pytorch_model.safetensors", "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.safetensors", Some(334_643_268)) ] }),
        other => Err(RegistryError::UnknownModel(other.into())),
    }
}

fn download_file(client: Client, source: SourceFile, output: &Path, progress: Arc<MultiProgress>) -> Result<DownloadFile> {
    let destination = output.join(&source.path);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let existing = inspect_component(output, &source);
    let mut offset = existing.actual_bytes.unwrap_or(0);
    if existing.complete {
        return Ok(DownloadFile {
            path: source.path,
            downloaded_bytes: offset,
            expected_bytes: source.expected_bytes,
            complete: true,
        });
    }
    let mut request = client.get(&source.url);
    if offset > 0 {
        request = request.header(RANGE, format!("bytes={offset}-"));
    }
    let mut response = request.send()?;
    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE {
        let remote_size = response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|header| header.to_str().ok())
            .and_then(|value| value.rsplit_once('/'))
            .and_then(|(_, size)| size.parse::<u64>().ok());
        if remote_size.is_some_and(|size| size == offset) {
            return Ok(DownloadFile {
                path: source.path,
                downloaded_bytes: offset,
                expected_bytes: remote_size,
                complete: true,
            });
        }
        offset = 0;
        let cache_buster = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let separator = if source.url.contains('?') { '&' } else { '?' };
        response = client.get(format!("{}{separator}sd_resume_reset={cache_buster}", source.url)).send()?;
    }
    let response = response.error_for_status()?;
    let append = offset > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
    if !append {
        offset = 0;
    }
    let total = response.content_length().map(|len| len + offset).or(source.expected_bytes);
    let bar = progress.add(ProgressBar::new(total.unwrap_or(0)));
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg:50} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .map_err(|e| RegistryError::Io(std::io::Error::other(e)))?,
    );
    bar.set_position(offset);
    bar.set_message(source.path.clone());
    let mut output_file = OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&destination)?;
    let mut response: Response = response;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output_file.write_all(&buffer[..read])?;
        offset += read as u64;
        bar.set_position(offset);
    }
    bar.finish();
    Ok(DownloadFile {
        path: source.path,
        downloaded_bytes: offset,
        expected_bytes: total,
        complete: total.is_none_or(|size| size == offset),
    })
}

pub fn download(id: &str, output: impl AsRef<Path>) -> Result<DownloadMetadata> {
    let model = well_known(id)?;
    let output = output.as_ref().to_path_buf();
    fs::create_dir_all(&output)?;
    let bars = Arc::new(MultiProgress::new());
    bars.set_draw_target(indicatif::ProgressDrawTarget::stderr());
    let mut builder = Client::builder().timeout(Duration::from_secs(600));
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let mut value = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|error| RegistryError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, error)))?;
        value.set_sensitive(true);
        builder = builder.default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(AUTHORIZATION, value);
            headers
        });
    }
    let client = builder.build()?;
    let results: Arc<Mutex<Vec<Option<Result<DownloadFile>>>>> =
        Arc::new(Mutex::new((0..model.files.len()).map(|_| None).collect()));
    thread::scope(|scope| {
        for (index, source) in model.files.iter().cloned().enumerate() {
            let client = client.clone();
            let output = output.clone();
            let bars = bars.clone();
            let results = results.clone();
            scope.spawn(move || {
                let result = download_file(client, source, &output, bars);
                results.lock().expect("download results lock")[index] = Some(result);
            });
        }
    });
    let files: Vec<DownloadFile> = results
        .lock()
        .expect("download results lock")
        .drain(..)
        .map(|result| match result {
            Some(Ok(file)) => Ok(file),
            Some(Err(error)) => Err(error),
            None => Err(RegistryError::WorkerPanic),
        })
        .collect::<Result<_>>()?;
    let metadata = DownloadMetadata { model, files };
    write_manifest_atomic(&output, &metadata)?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd15_trusted_clip_length_matches_valid_diffusers_export() {
        let model = well_known("sd15").expect("sd15 source");
        let clip = model.files.iter().find(|file| file.path == "text_encoder/model.safetensors").expect("clip source");
        assert_eq!(clip.expected_bytes, Some(492_265_874));
    }

    #[test]
    fn missing_model_directory_has_structured_component_report() {
        let root = std::env::temp_dir().join(format!("sd-registry-missing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let report = verify("sd15", &root).expect("report");
        assert_eq!(report.components.len(), 4);
        assert!(!report.complete());
        assert!(report.components.iter().all(|component| !component.exists && component.detail == "missing"));
    }

    #[test]
    fn verify_atomically_migrates_a_stale_manifest_to_trusted_metadata() {
        let root = std::env::temp_dir().join(format!("sd-registry-migrate-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("temporary root");
        let mut stale_model = well_known("sd15").expect("sd15 source");
        stale_model.files[1].expected_bytes = Some(643_392_057);
        let stale = DownloadMetadata { model: stale_model, files: vec![] };
        fs::write(root.join(".sd-download.json"), serde_json::to_vec(&stale).expect("serialize stale manifest")).expect("write stale manifest");
        let report = verify("sd15", &root).expect("migrate manifest");
        assert!(report.legacy_manifest_migrated);
        let refreshed: DownloadMetadata = serde_json::from_slice(&fs::read(root.join(".sd-download.json")).expect("read refreshed manifest")).expect("parse refreshed manifest");
        assert_eq!(refreshed.model.files[1].expected_bytes, Some(492_265_874));
        assert!(!root.join(".sd-download.json.tmp").exists());
        let _ = fs::remove_dir_all(root);
    }
}
