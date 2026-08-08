//! `sd`: local Stable Diffusion generation and model-maintenance tool.
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion::{self, vae::AutoEncoderKL};
use clap::{Parser, Subcommand, ValueEnum};
use rand::Rng;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(name = "sd", about = "Lightweight local Stable Diffusion tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Generate(GenerateArgs),
    /// Download a well-known model with HTTP range resume support.
    Download {
        /// Registry model id, for example `sd15`.
        model: String,
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
    Serve {
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = Family::Sd15)]
        family: Family,
        #[arg(long)]
        output_dir: Option<PathBuf>,
        #[arg(long)]
        pages_dir: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1:3000")]
        address: std::net::SocketAddr,
    },
    Inspect {
        model: PathBuf,
    },
    BakeVae {
        checkpoint: PathBuf,
        vae: PathBuf,
        output: PathBuf,
    },
    /// Verify a downloaded model directory against trusted registry metadata.
    Verify {
        /// Registry model id, for example `sd15`.
        model: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
}
#[derive(Clone, Copy, ValueEnum)]
enum Family {
    Sd15,
    Sd21,
    Sdxl,
}
#[derive(Clone, Parser)]
struct GenerateArgs {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value = "")]
    negative_prompt: String,
    #[arg(long, value_enum, default_value_t = Family::Sd15)]
    family: Family,
    #[arg(long, default_value_t = 512)]
    width: usize,
    #[arg(long, default_value_t = 512)]
    height: usize,
    #[arg(long, default_value_t = 20)]
    steps: usize,
    #[arg(long, default_value_t = 7.5)]
    cfg_scale: f64,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long = "lora", value_parser = parse_lora)]
    loras: Vec<diffusion_types::LoraSpec>,
    #[arg(long = "embedding")]
    embeddings: Vec<PathBuf>,
    #[arg(long)]
    conditioning: Option<PathBuf>,
    #[arg(long = "controlnet")]
    controlnets: Vec<String>,
    #[arg(long)]
    annotator: Option<String>,
    #[arg(long, default_value = "output.png")]
    output: PathBuf,
}
fn parse_lora(value: &str) -> std::result::Result<diffusion_types::LoraSpec, String> {
    let (path, weight) = value.rsplit_once(':').ok_or("LoRA must use PATH:WEIGHT")?;
    Ok(diffusion_types::LoraSpec { path: PathBuf::from(path), weight: weight.parse().map_err(|_| "invalid LoRA weight".to_string())? })
}
fn path(root: &Path, relative: &str) -> Result<PathBuf> {
    let p = root.join(relative);
    p.exists().then_some(p.clone()).ok_or_else(|| anyhow::anyhow!("required model asset missing: {}", p.display()))
}
fn executable_dir() -> Result<PathBuf> {
    std::env::current_exe()?.parent().map(Path::to_path_buf).context("sd executable has no parent directory")
}
fn tokens(tokenizer: &Tokenizer, text: &str, length: usize, pad: u32, device: &Device) -> Result<Tensor> {
    let mut ids = tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    if ids.len() > length {
        bail!("prompt exceeds {length} CLIP tokens")
    }
    ids.resize(length, pad);
    Ok(Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?)
}
fn clip_embeddings(
    clip_config: &stable_diffusion::clip::Config,
    tokenizer_path: PathBuf,
    weights: PathBuf,
    prompt: &str,
    negative: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let pad = *tokenizer
        .get_vocab(true)
        .get(clip_config.pad_with.as_deref().unwrap_or("<|endoftext|>"))
        .context("tokenizer has no pad token")?;
    let model = stable_diffusion::build_clip_transformer(clip_config, weights, device, DType::F32)?;
    let positive = model.forward(&tokens(&tokenizer, prompt, clip_config.max_position_embeddings, pad, device)?)?;
    let negative = model.forward(&tokens(&tokenizer, negative, clip_config.max_position_embeddings, pad, device)?)?;
    Ok(Tensor::cat(&[negative, positive], 0)?.to_dtype(dtype)?)
}
fn embeddings(
    config: &stable_diffusion::StableDiffusionConfig,
    model_dir: &Path,
    prompt: &str,
    negative: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let first = clip_embeddings(
        &config.clip,
        path(model_dir, "tokenizer/tokenizer.json")?,
        path(model_dir, "text_encoder/model.safetensors")?,
        prompt,
        negative,
        device,
        dtype,
    )?;
    match &config.clip2 {
        None => Ok(first),
        Some(clip2) => {
            let second = clip_embeddings(
                clip2,
                path(model_dir, "tokenizer_2/tokenizer.json")?,
                path(model_dir, "text_encoder_2/model.safetensors")?,
                prompt,
                negative,
                device,
                dtype,
            )?;
            Ok(Tensor::cat(&[first, second], candle_core::D::Minus1)?)
        }
    }
}
fn save(vae: &AutoEncoderKL, latents: &Tensor, scale: f64, output: &Path) -> Result<()> {
    let images = vae.decode(&(latents / scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?.clamp(0f32, 1.)?;
    let image = (images.get(0)? * 255.)?.to_dtype(DType::U8)?;
    candle_examples::save_image(&image, output)?;
    Ok(())
}
fn generate(args: GenerateArgs) -> Result<u64> {
    if args.width % 64 != 0 || args.height % 64 != 0 {
        bail!("width and height must be multiples of 64")
    }
    let device = Device::Cpu;
    let default_model = match args.family {
        Family::Sd15 => "sd15",
        Family::Sd21 => "sd21",
        Family::Sdxl => "sdxl",
    };
    let model_dir = args.model_dir.clone().unwrap_or(executable_dir()?.join("models").join(default_model));
    if !args.loras.is_empty() || !args.embeddings.is_empty() || !args.controlnets.is_empty() || args.annotator.is_some() {
        bail!("LoRA, textual inversion and ControlNet inputs are parsed but not yet executable; refusing to ignore them");
    }
    let dtype = DType::F32;
    let config = match args.family {
        Family::Sd15 => stable_diffusion::StableDiffusionConfig::v1_5(None, Some(args.height), Some(args.width)),
        Family::Sd21 => stable_diffusion::StableDiffusionConfig::v2_1(None, Some(args.height), Some(args.width)),
        Family::Sdxl => stable_diffusion::StableDiffusionConfig::sdxl(None, Some(args.height), Some(args.width)),
    };
    let seed = args.seed.unwrap_or_else(|| rand::rng().random());
    let unet = path(&model_dir, "unet/diffusion_pytorch_model.safetensors")?;
    let vae = path(&model_dir, "vae/diffusion_pytorch_model.safetensors")?;
    let text = if let Some(conditioning) = &args.conditioning {
        let tensors = diffusion_extensions::load_conditioning(conditioning, &device)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let positive = tensors.get("prompt_embeds").context("conditioning missing prompt_embeds")?;
        let negative = tensors.get("negative_prompt_embeds").context("conditioning missing negative_prompt_embeds")?;
        if positive.dims() != negative.dims() || positive.rank() != 3 || positive.dim(0)? != 1 {
            bail!("conditioning prompt tensors must have matching shape [1, sequence, hidden]");
        }
        let hidden = positive.dim(2)?;
        let expected = match args.family { Family::Sd15 => 768, Family::Sd21 => 1024, Family::Sdxl => 2048 };
        if hidden != expected { bail!("conditioning hidden size {hidden} does not match model family (expected {expected})"); }
        Tensor::cat(&[negative, positive], 0)?.to_dtype(dtype)?
    } else {
        embeddings(&config, &model_dir, &args.prompt, &args.negative_prompt, &device, dtype)?
    };
    let vae = config.build_vae(vae, &device, dtype)?;
    let unet = config.build_unet(unet, &device, 4, false, dtype)?;
    let mut scheduler = config.build_scheduler(args.steps)?;
    let mut latents = (Tensor::randn(0f32, 1f32, (1, 4, args.height / 8, args.width / 8), &device)?
        * scheduler.init_noise_sigma())?
    .to_dtype(dtype)?;
    let timesteps = scheduler.timesteps().to_vec();
    for (index, t) in timesteps.into_iter().enumerate() {
        let input = scheduler.scale_model_input(Tensor::cat(&[&latents, &latents], 0)?, t)?;
        let noise = unet.forward(&input, t as f64, &text)?;
        let pair = noise.chunk(2, 0)?;
        let guided = (&pair[0] + ((&pair[1] - &pair[0])? * args.cfg_scale)?)?;
        latents = scheduler.step(&guided, t, &latents)?;
        eprintln!("step {}/{}", index + 1, args.steps);
    }
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save(&vae, &latents, 0.18215, &args.output)?;
    println!("seed={seed} image={}", args.output.display());
    Ok(seed)
}
#[derive(Clone)]
struct LocalPipeline {
    model_dir: PathBuf,
    output_dir: PathBuf,
    family: Family,
}
impl diffusion_types::DiffusionPipeline for LocalPipeline {
    fn generate(
        &self,
        request: &diffusion_types::GenerationRequest,
        progress: &dyn diffusion_types::ProgressSink,
        cancel: &diffusion_types::CancellationToken,
    ) -> diffusion_types::Result<diffusion_types::GenerationResult> {
        if cancel.is_cancelled() {
            return Err(diffusion_types::DiffusionError::Cancelled);
        }
        std::fs::create_dir_all(&self.output_dir).map_err(diffusion_types::DiffusionError::Io)?;
        let output = self.output_dir.join(format!("{}.png", uuid::Uuid::new_v4()));
        let args = GenerateArgs {
            model_dir: Some(self.model_dir.clone()),
            prompt: request.prompt.clone(),
            negative_prompt: request.negative_prompt.clone().unwrap_or_default(),
            family: self.family,
            width: request.width as usize,
            height: request.height as usize,
            steps: request.steps as usize,
            cfg_scale: request.guidance_scale as f64,
            seed: request.seed,
            loras: request.loras.clone(),
            embeddings: request.embeddings.iter().map(|e| e.path.clone()).collect(),
            conditioning: request.conditioning.as_ref().map(|c| c.path.clone()),
            controlnets: request.controlnets.iter().map(|c| format!("{}:{}:{}:{}:{}", c.model.display(), c.image.display(), c.weight, c.start, c.end)).collect(),
            annotator: request.annotator.as_ref().map(|a| a.name.clone()),
            output: output.clone(),
        };
        let seed = generate(args).map_err(|error| diffusion_types::DiffusionError::Model(error.to_string()))?;
        progress.step(request.steps, request.steps);
        let image = image::open(output).map_err(|error| diffusion_types::DiffusionError::Model(error.to_string()))?;
        Ok(diffusion_types::GenerationResult { images: vec![image], seed })
    }
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Generate(args) => {
            generate(args)?;
            Ok(())
        }
        Command::Download { model, output_dir } => {
            let output_dir = output_dir.unwrap_or(executable_dir()?.join("models"));
            let metadata = diffusion_registry::download(&model, output_dir.join(&model))?;
            println!(
                "{} download metadata written; complete={}",
                metadata.model.display_name,
                metadata.files.iter().all(|file| file.complete)
            );
            Ok(())
        }
        Command::Serve { model_dir, family, output_dir, pages_dir, address } => {
            let executable_dir = executable_dir()?;
            let default_model = match family {
                Family::Sd15 => "sd15",
                Family::Sd21 => "sd21",
                Family::Sdxl => "sdxl",
            };
            let model_dir = model_dir.unwrap_or_else(|| executable_dir.join("models").join(default_model));
            let output_dir = output_dir.unwrap_or_else(|| executable_dir.join("outputs"));
            let pages_dir = pages_dir.unwrap_or_else(|| executable_dir.join("pages"));
            let pipeline = Arc::new(LocalPipeline { model_dir, output_dir, family });
            let state = diffusion_server::AppState {
                tasks: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
                pipeline: Some(pipeline),
            };
            println!("sd listening on http://{address}");
            tokio::runtime::Runtime::new()?.block_on(diffusion_server::serve(state, address, pages_dir))?;
            Ok(())
        }
        Command::Inspect { model } => {
            println!("{:#?}", diffusion_models::inspect(&model)?);
            Ok(())
        }
        Command::BakeVae { checkpoint, vae, output } => {
            diffusion_tools::bake_vae(&checkpoint, &vae, &output).map_err(anyhow::Error::msg)
        }
        Command::Verify { model, model_dir } => {
            let model_dir = match model_dir {
                Some(model_dir) => model_dir,
                None => executable_dir()?.join("models").join(&model),
            };
            let report = diffusion_registry::verify(&model, &model_dir).map_err(|error| anyhow::anyhow!(error.to_string()))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if report.complete() { Ok(()) } else { bail!("model integrity verification failed") }
        }
    }
}
