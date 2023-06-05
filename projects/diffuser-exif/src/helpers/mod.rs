use std::fs::{self, File};

use img_parts::png::Png;
use img_parts::{Bytes, ImageEXIF, ImageICC};
use image::ImageError;


fn read_exif_from_png(image_path: &str) -> Result<(), ImageError> {
    // 读取图片文件
    let decoder = png::Decoder::new(File::open(image_path)?);
    let mut reader = decoder.read_info().unwrap();
    // If the text chunk is before the image data frames, `reader.info()` already contains the text.
    for text_chunk in &reader.info().uncompressed_latin1_text {
        println!("{:?}", text_chunk.keyword); // Prints the keyword
        println!("{:#?}", text_chunk); // Prints out the text chunk.
        // To get the uncompressed text, use the `get_text` method.
        // println!("{:}", text_chunk);
    }
    Ok(())
}

#[test]
fn main() {
    let exif = read_exif_from_png(r#"C:\Users\Aster\Downloads\c62755dc02d1bb48608588b44a3a7fcfc838d36b8f01bf845fc7cc46c16bfe1e.png"#).expect("Failed to read EXIF from the PNG image");
    // for f in exif.fields() {
    //     println!("{} {} {}",
    //              f.tag, f.ifd_num, f.display_value().with_unit(&exif));
    // }
}


use sha2::{Sha256, Digest};
use std::collections::HashSet;

const TARGET_PREFIX: &str = "114514";
const CHARSET: &str = "哼嗯啊呃嗷呜哦";

fn number_to_string(num: usize) -> String {
    let mut result = String::with_capacity(16); // 预分配内存
    let mut remaining = num;
    while remaining > 0 {
        let index = remaining % CHARSET.len();
        result.insert(0, CHARSET.chars().nth(index).unwrap());
        remaining /= CHARSET.len();
    }
    result
}

#[test]
fn main2() {
    let mut solutions = HashSet::new();
    let options = "哼啊";

    for length in 21..=24 {
        generate_combinations(options, length, &mut solutions);
        for solution in &solutions {
            if is_valid_solution(solution) {
                println!("{}", solution);
                // return;
            }
        }
        solutions.clear();
    }

    println!("No solution found.");
}

fn generate_combinations(options: &str, length: usize, solutions: &mut HashSet<String>) {
    if length == 0 {
        solutions.insert(String::new());
        return;
    }

    for c in options.chars() {
        let mut prefix = String::new();
        prefix.push(c);
        for combination in generate_combinations_helper(options, length - 1, &prefix) {
            solutions.insert(combination);
        }
    }
}

fn generate_combinations_helper(options: &str, length: usize, prefix: &str) -> Vec<String> {
    if length == 0 {
        return vec![prefix.to_string()];
    }

    let mut results = Vec::new();
    for c in options.chars() {
        let mut new_prefix = prefix.to_string();
        new_prefix.push(c);
        for combination in generate_combinations_helper(options, length - 1, &new_prefix) {
            results.push(combination);
        }
    }
    results
}

fn is_valid_solution(input: &str) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    let hex_result = hex_bytes(&result);
    hex_result.starts_with(TARGET_PREFIX)
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}