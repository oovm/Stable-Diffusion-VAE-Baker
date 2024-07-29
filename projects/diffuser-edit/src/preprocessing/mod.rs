use std::error::Error;
use std::fs;
use std::path::Path;
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgb, RgbImage};
use walkdir::WalkDir;

pub struct ImageProcessing {
    pub crop_alpha: bool,
    pub erase_alpha: bool,
    pub delete_source: bool,
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

                    let handle = std::thread::spawn(|| {
                        // 异步执行的代码
                        println!("Hello from spawned thread!");
                    })
                        .join();

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

