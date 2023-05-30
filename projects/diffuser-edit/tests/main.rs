use std::path::Path;
use diffuser_edit::{bake_vae_by_path};


#[test]
fn ready() {
    println!("it works!")
}


#[test]
fn run() -> candle_core::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = here.join("tests/DeepOcean-720000.safetensors");
    let vae = here.join("tests/NyanMix.vae.pt");
    bake_vae_by_path(&model, &vae)
}

#[test]
fn run2() -> candle_core::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = here.join("tests/Shiny-720000.safetensors");
    let vae = here.join("tests/kl-f8-anime2.vae.safetensors");
    bake_vae_by_path(&model, &vae)
}

