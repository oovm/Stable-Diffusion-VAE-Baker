use std::collections::HashMap;
use diffuser_edit::{bake_vae_by_path, ImageProcessing};
use std::path::Path;
use candle_core::{Device, Tensor};

#[test]
fn ready() {
    println!("it works!")
}

#[test]
#[ignore]
fn run() -> candle_core::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();
    bake_deep_ocean()?;
    // bake_shiny_sky()
    Ok(())
}

fn bake_deep_ocean() -> candle_core::Result<()> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = here.join("tests/hassakuXLHentai_v13BetterEyesVersion.safetensors");
    let vae = here.join("tests/xlVAEC_f1.safetensors");
    bake_vae_by_path(&model, &vae)
}

// fn bake_shiny_sky() -> candle_core::Result<()> {
//     let here = Path::new(env!("CARGO_MANIFEST_DIR"));
//     let model = here.join("tests/ShinySky-640000.safetensors");
//     let vae = here.join("tests/kl-f8-anime2.vae.safetensors");
//     bake_vae_by_path(&model, &vae)
// }

#[test]
fn text_em() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/badhandv4.pt");
    let array: HashMap<String, Tensor> = candle_core::pickle::read_all_with_key(path, None).unwrap().into_iter().collect();
    for (k, v) in array.iter() {
        println!("{}: {:?}", k, v.dtype());
    }
}

// #[test]
// fn main() -> Result<(), Box<dyn Error>> {
//     let safetensors_file: &[u8] = include_bytes!(r#"C:\Users\Aster\Downloads\gatomon.safetensors"#);
//
//     let (safetensors, meta) = SafeTensors::read_metadata(safetensors_file)?;
//
//     println!("SafeTensors Metadata:");
//     for (key, metadata) in meta.metadata().clone().unwrap() {
//         println!("{}: {:?}", key, metadata);
//     }
//
//     Ok(())
// }

#[test]
fn main2() {
    let dir = Path::new(r#"C:\Users\Administrator\Downloads"#);
    let step = ImageProcessing {
        crop_alpha: true,
        erase_alpha: true,
        delete_source: true,
    };
    if let Err(e) = step.convert_directory(&dir) {
        eprintln!("Error processing directory: {:?}", e);
    }
}
