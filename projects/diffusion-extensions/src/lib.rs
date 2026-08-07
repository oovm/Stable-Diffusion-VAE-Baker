//! Optional image annotators and extension configuration.
use diffusion_types::{DiffusionError, Result};
use image::{DynamicImage, GrayImage, Luma};
pub trait Annotator: Send + Sync {
    fn annotate(&self, image: &DynamicImage) -> Result<DynamicImage>;
}
#[derive(Debug, Default, Clone, Copy)]
pub struct Canny;
impl Annotator for Canny {
    fn annotate(&self, image: &DynamicImage) -> Result<DynamicImage> {
        let gray = image.to_luma8();
        let mut out = GrayImage::new(gray.width(), gray.height());
        for y in 1..gray.height().saturating_sub(1) {
            for x in 1..gray.width().saturating_sub(1) {
                let gx = i32::from(gray.get_pixel(x + 1, y)[0]) - i32::from(gray.get_pixel(x - 1, y)[0]);
                let gy = i32::from(gray.get_pixel(x, y + 1)[0]) - i32::from(gray.get_pixel(x, y - 1)[0]);
                let v = ((gx.abs() + gy.abs()).min(255)) as u8;
                out.put_pixel(x, y, Luma([if v > 80 { 255 } else { 0 }]));
            }
        }
        Ok(DynamicImage::ImageLuma8(out))
    }
}
pub fn unavailable(name: &str) -> Result<()> {
    Err(DiffusionError::Model(format!("annotator `{name}` requires an external Candle model")))
}
