//! Well-known local model sources, parallel downloads, and resumable transfers.
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::{
    StatusCode,
    blocking::{Client, Response},
    header::{AUTHORIZATION, CONTENT_RANGE, HeaderValue, RANGE},
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceFile {
    pub path: String,
    pub url: String,
    pub expected_bytes: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
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

fn source(path: &str, url: &str, expected_bytes: Option<u64>) -> SourceFile {
    SourceFile { path: path.into(), url: url.into(), expected_bytes }
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
    let mut offset = destination.metadata().map(|meta| meta.len()).unwrap_or(0);
    if source.expected_bytes.is_some_and(|size| offset == size) {
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
    fs::write(output.join(".sd-download.json"), serde_json::to_vec_pretty(&metadata)?)?;
    Ok(metadata)
}
