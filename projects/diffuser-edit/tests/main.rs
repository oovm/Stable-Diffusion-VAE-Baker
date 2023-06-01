use diffuser_edit::bake_vae_by_path;
use std::path::Path;

#[test]
fn ready() {
    println!("it works!")
}

#[test]
#[ignore]
fn run() -> candle_core::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();
    bake_deep_ocean()?;
    bake_shiny_sky()
}

fn bake_deep_ocean() -> candle_core::Result<()> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = here.join("tests/DeepOcean-720000.safetensors");
    let vae = here.join("tests/kl-f8-anime2.vae.safetensors");
    bake_vae_by_path(&model, &vae)
}

fn bake_shiny_sky() -> candle_core::Result<()> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = here.join("tests/ShinySky-640000.safetensors");
    let vae = here.join("tests/kl-f8-anime2.vae.safetensors");
    bake_vae_by_path(&model, &vae)
}
