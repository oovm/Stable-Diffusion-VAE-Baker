use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::Path};
use std::error::Error;
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgb, RgbImage};
use std::fs;


use walkdir::WalkDir;

pub fn load_model(path: &Path) -> candle_core::Result<HashMap<String, Tensor>> {
    let tensors = match path.extension() {
        Some(s) if s.eq("pt") => candle_core::pickle::read_all_with_key(path, None)?.into_iter().collect(),
        Some(s) if s.eq("ckpt") => {
            unimplemented!("loading `ckpt` checkpoints is not yet supported")
        }
        _ => candle_core::safetensors::load(path, &Device::Cpu)?,
    };
    Ok(tensors)
}

/// Quantize the float tensors to f16.
pub fn quantize_f16(checkpoint: &mut HashMap<String, Tensor>) -> candle_core::Result<()> {
    for (k, v) in checkpoint.iter_mut() {
        match v.dtype() {
            DType::U8 => {}
            DType::U32 => {}
            DType::I64 => {}
            DType::BF16 => {}
            DType::F16 => {}
            DType::F32 => {
                tracing::info!("    Quantize: f32 `{}` to f16", k);
                *v = v.to_dtype(DType::BF16)?
            }
            DType::F64 => {
                tracing::info!("    Quantize: f64 `{}` to f16", k);
                *v = v.to_dtype(DType::BF16)?
            }
        }
    }
    Ok(())
}


fn convert_image(input_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Load the image
    let img = image::open(input_path)?;
    let img = erase_alpha(&img)?;

    // Prepare the output path
    let mut output_path = input_path.to_path_buf();
    output_path.set_extension("jpg");

    // Save the image as JPEG with 95% quality
    let mut output_file = fs::File::create(&output_path)?;
    img.write_to(&mut output_file, ImageFormat::Jpeg)?;

    // Remove the original file
    fs::remove_file(input_path)?;

    println!("Converted and deleted: {:?}", input_path);

    Ok(())
}

pub fn process_directory(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            match path.extension().and_then(|s| s.to_str()) {
                Some("webp") | Some("avif") | Some("gif") | Some("png") => {
                    if let Err(e) = convert_image(path) {
                        eprintln!("Failed to convert {:?}: {:?}", path, e);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn crop_alpha(mut image: DynamicImage) -> Result<DynamicImage, Box<dyn Error>> {
    // 裁剪掉四周的空白像素
    let (width, height) = image.dimensions();
    let mut left = width;
    let mut right = 0;
    let mut top = height;
    let mut bottom = 0;

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            if pixel[3] > 0 {
                left = left.min(x);
                right = right.max(x);
                top = top.min(y);
                bottom = bottom.max(y);
            }
        }
    }

    let image = image.crop(left, top, right - left + 1, bottom - top + 1);
    Ok(image)
}


fn erase_alpha(image: &DynamicImage) -> Result<RgbImage, Box<dyn Error>> {
    // 将透明背景变为黑色
    let mut output_img = ImageBuffer::new(image.width(), image.height());
    // 遍历裁剪后的图像,并计算半透明像素在黑色背景下的颜色值
    for (x, y, pixel) in image.pixels() {
        let alpha = pixel[3] as f32 / 255.0;
        let r = (pixel[0] as f32 * alpha + 0.0 * (1.0 - alpha)) as u8;
        let g = (pixel[1] as f32 * alpha + 0.0 * (1.0 - alpha)) as u8;
        let b = (pixel[2] as f32 * alpha + 0.0 * (1.0 - alpha)) as u8;
        let new_pixel = Rgb::from([r, g, b]);
        output_img.put_pixel(x, y, new_pixel);
    }

    return Ok(output_img);
}

