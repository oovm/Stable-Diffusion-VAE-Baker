use std::error::Error;
use std::ffi::OsStr;
use std::fs::File;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, SendError};
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageError, ImageFormat, Rgb, RgbImage};
use walkdir::WalkDir;


#[derive(Copy, Clone, Debug)]
pub struct ImageProcessing {
    pub crop_alpha: bool,
    pub erase_alpha: bool,
    pub delete_source: bool,
}

impl ImageProcessing {
    pub fn convert_directory(&self, path: &Path) {
        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            let extension = match path.extension() {
                Some(s) => { s.to_ascii_lowercase() }
                None => { continue }
            };
            match extension.to_str() {
                Some("webp") | Some("avif") | Some("gif") | Some("png") | Some("jpg_large") | Some("jpg_small") | Some("jpg") => {
                    let this = self.clone();
                    let path = path.to_path_buf();
                    this.convert_path(&path).unwrap();
                }
                _ => {}
            }
        }
    }

    pub fn convert_path<P: AsRef<Path>>(&self, file: P) -> Result<(), ImageError> {
        let path = file.as_ref();
        // Load the image
        let img = image::open(path)?;
        let img = erase_alpha(&img)?;
        // Prepare the output path
        let mut output_path = path.to_path_buf();
        output_path.set_extension("jpeg");
        // Save the image as JPEG with 95% quality
        let mut output_file = File::create(&output_path)?;
        img.write_to(&mut output_file, ImageFormat::Jpeg)?;
        if self.delete_source {
            // Remove the original file
            std::fs::remove_file(path)?;
            println!("Converted and deleted: {:?}", path);
        } else {
            println!("Converted: {:?}", path);
        }
        Ok(())
    }
}


fn crop_alpha(mut image: DynamicImage) -> Result<DynamicImage, ImageError> {
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

fn erase_alpha(image: &DynamicImage) -> Result<RgbImage, ImageError> {
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

