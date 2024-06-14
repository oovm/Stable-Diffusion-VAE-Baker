use std::path::Path;
use candle_core::safetensors::{save};
use diffuser_edit::bake_vae;
use diffuser_edit::helpers::load_model;

#[test]
fn ready() {
    println!("it works!")
}


#[test]
fn run() -> candle_core::Result<()> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = here.join("tests/DeepOcean-720000.safetensors");
    let vae = here.join("tests/DeepOcean.vae.pt");
    save_baked(&model, &vae)
}

#[test]
fn run2() -> candle_core::Result<()> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = here.join("tests/Shiny-720000.safetensors");
    let vae = here.join("tests/kl-f8-anime2.vae.safetensors");
    save_baked(&model, &vae)
}


fn save_baked(model_file_path: &Path, vae_file_path: &Path) -> candle_core::Result<()> {
    let checkpoint = load_model(model_file_path)?;
    let vae = load_model(vae_file_path)?;
    let output = bake_vae(checkpoint, &vae)?;
    let name = format!("{}-{}.safetensors", model_file_path.file_stem().unwrap().to_str().unwrap(), vae_file_path.file_stem().unwrap().to_str().unwrap());
    save(&output, name)
}


