use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs::File;
use diffuser_edit::{bake_vae_by_path, ImageProcessing};
use std::path::{Path, PathBuf};
use candle_core::{DType, Device, Tensor};
use candle_core::safetensors::Load;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

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


pub fn sign_model(path: &str, path2: &Path) -> Result<(), Box<dyn Error>> {
    let safetensors_file = std::fs::read(path)?;
    let safetensors = SafeTensors::deserialize(&safetensors_file)?;
    let (_, meta) = SafeTensors::read_metadata(&safetensors_file)?;
    let mut meta = match meta.metadata() {
        Some(s) => { s.clone() }
        None => { Default::default() }
    };
    meta.insert("ss_author".to_string(), "https://civitai.com/user/XEZ".to_string());
    safetensors::serialize_to_file(safetensors.tensors(), &Some(meta), path2)?;
    //
    // println!("SafeTensors Metadata:");
    // for (key, metadata) in meta.metadata().clone().unwrap() {
    //     println!("{}: {:?}", key, metadata);
    // }
    Ok(())
}

#[test]
fn main3() {
    let dir = sign_model(
        r#"C:\Users\Aster\Downloads\realisticFreedom3_auroraV09.safetensors"#,
        &Path::new(r#"C:\Users\Aster\Downloads\XE_UNREAL_SD3.safetensors"#),
    );
    // step.convert_directory(&dir)
}


#[test]
fn main2() {
    let dir = Path::new(r#"C:\Users\Aster\Downloads"#);
    let step = ImageProcessing {
        crop_alpha: true,
        erase_alpha: true,
        delete_source: true,
    };
    step.convert_directory(&dir)
}
