use std::error::Error;
use std::path::{Path};
use safetensors::SafeTensors;

#[test]
fn ready() {
    println!("it works!")
}

// fn bake_shiny_sky() -> candle_core::Result<()> {
//     let here = Path::new(env!("CARGO_MANIFEST_DIR"));
//     let model = here.join("tests/ShinySky-640000.safetensors");
//     let vae = here.join("tests/kl-f8-anime2.vae.safetensors");
//     bake_vae_by_path(&model, &vae)
// }

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
#[ignore]
fn main3() {
    let dir = sign_model(
        r#"C:\Users\Aster\Downloads\realisticFreedom3_auroraV09.safetensors"#,
        &Path::new(r#"C:\Users\Aster\Downloads\XE_UNREAL_SD3.safetensors"#),
    );
    // step.convert_directory(&dir)
}
