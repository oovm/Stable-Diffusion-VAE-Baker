use exif::Exif;
use image::ImageError;

fn read_exif_from_png(image_path: &str) -> Result<Exif, ImageError> {
    // 读取图片文件
    let file = std::fs::read(image_path)?;

    let exifreader = exif::Reader::new();

    let exif = exif::parse_exif(&file).unwrap();

    Ok(exif)
}

#[test]
fn main() {
    let exif = read_exif_from_png(r#"C:\Users\Aster\Downloads\c62755dc02d1bb48608588b44a3a7fcfc838d36b8f01bf845fc7cc46c16bfe1e.png"#).expect("Failed to read EXIF from the PNG image");
    for f in exif.fields() {
        println!("{} {} {}",
                 f.tag, f.ifd_num, f.display_value().with_unit(&exif));
    }
}