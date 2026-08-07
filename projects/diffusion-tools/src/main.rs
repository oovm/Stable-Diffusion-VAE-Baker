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
        #[arg(long, default_value = "models")]
        output_dir: PathBuf,
    },
    Serve {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "outputs")]
        output_dir: PathBuf,
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
}
#[derive(Clone, Copy, ValueEnum)]
enum Family {
    Sd15,
    Sdxl,
}
#[derive(Clone, Parser)]
struct GenerateArgs {
    #[arg(long)]
    model_dir: PathBuf,
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
    #[arg(long, default_value = "output.png")]
    output: PathBuf,
}
fn path(root: &Path, relative: &str) -> Result<PathBuf> {
    let p = root.join(relative);
    p.exists().then_some(p.clone()).ok_or_else(|| anyhow::anyhow!("required model asset missing: {}", p.display()))
}
fn tokens(tokenizer: &Tokenizer, text: &str, length: usize, pad: u32, device: &Device) -> Result<Tensor> {
    let mut ids = tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    if ids.len() > length {
        bail!("prompt exceeds {length} CLIP tokens")
    }
    ids.resize(length, pad);
    Ok(Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?)
}
fn embeddings(
    config: &stable_diffusion::StableDiffusionConfig,
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
        .get(config.clip.pad_with.as_deref().unwrap_or("<|endoftext|>"))
        .context("tokenizer has no pad token")?;
    let model = stable_diffusion::build_clip_transformer(&config.clip, weights, device, DType::F32)?;
    let positive = model.forward(&tokens(&tokenizer, prompt, config.clip.max_position_embeddings, pad, device)?)?;
    let negative = model.forward(&tokens(&tokenizer, negative, config.clip.max_position_embeddings, pad, device)?)?;
    Ok(Tensor::cat(&[negative, positive], 0)?.to_dtype(dtype)?)
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
    let dtype = DType::F32;
    let config = match args.family {
        Family::Sd15 => stable_diffusion::StableDiffusionConfig::v1_5(None, Some(args.height), Some(args.width)),
        Family::Sdxl => bail!("SDXL is not yet wired into the sd CLI; use --family sd15"),
    };
    let seed = args.seed.unwrap_or_else(|| rand::rng().random());
    device.set_seed(seed)?;
    let tokenizer = path(&args.model_dir, "tokenizer/tokenizer.json")?;
    let clip = path(&args.model_dir, "text_encoder/model.safetensors")?;
    let unet = path(&args.model_dir, "unet/diffusion_pytorch_model.safetensors")?;
    let vae = path(&args.model_dir, "vae/diffusion_pytorch_model.safetensors")?;
    let text = embeddings(&config, tokenizer, clip, &args.prompt, &args.negative_prompt, &device, dtype)?;
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
    save(&vae, &latents, 0.18215, &args.output)?;
    println!("seed={seed} image={}", args.output.display());
    Ok(seed)
}
#[derive(Clone)]
struct LocalPipeline {
    model_dir: PathBuf,
    output_dir: PathBuf,
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
            model_dir: self.model_dir.clone(),
            prompt: request.prompt.clone(),
            negative_prompt: request.negative_prompt.clone().unwrap_or_default(),
            family: Family::Sd15,
            width: request.width as usize,
            height: request.height as usize,
            steps: request.steps as usize,
            cfg_scale: request.guidance_scale as f64,
            seed: request.seed,
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
            let metadata = diffusion_registry::download(&model, output_dir.join(&model))?;
            println!(
                "{} download metadata written; complete={}",
                metadata.model.display_name,
                metadata.files.iter().all(|file| file.complete)
            );
            Ok(())
        }
        Command::Serve { model_dir, output_dir, address } => {
            let pipeline = Arc::new(LocalPipeline { model_dir, output_dir });
            let state = diffusion_server::AppState {
                tasks: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
                pipeline: Some(pipeline),
            };
            println!("sd listening on http://{address}");
            tokio::runtime::Runtime::new()?.block_on(diffusion_server::serve(state, address))?;
            Ok(())
        }
        Command::Inspect { model } => {
            println!("{:#?}", diffusion_models::inspect(&model)?);
            Ok(())
        }
        Command::BakeVae { checkpoint, vae, output } => {
            diffusion_tools::bake_vae(&checkpoint, &vae, &output).map_err(anyhow::Error::msg)
        }
    }
}
