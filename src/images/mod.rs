use image::*;
use image::GrayImage;
use std::fs::*;



// load image file from disk
pub fn load_image(path: &str) -> Vec<Vec<f64>> {
    let img = image::open(path).unwrap();
    let (width, height) = img.dimensions();
    let mut pixels = Vec::new();
    for y in 0..height {
        let mut row = Vec::new();
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as f64 / 255.0;
            let g = pixel[1] as f64 / 255.0;
            let b = pixel[2] as f64 / 255.0;
            row.push(r);
            row.push(g);
            row.push(b);
        }
        pixels.push(row);
    }
    pixels
}

fn is_within_range(img: &GrayImage) -> bool {
    for pixel in img.pixels() {
        if pixel[0] < 0 || pixel[0] > 255 {
            return false;
        }
    }
    true
}

fn is_8_bit_grayscale(img_path: &std::path::Path) -> bool {
    match image::open(img_path) {
        Ok(img) => {
            let color = img.color();
            match color {
                ColorType::L8 => true,  // L8 denotes 8-bit grayscale
                _ => false,
            }
        }
        Err(_) => false,
    }
}


pub fn check_image() {
    let directories = ["./mnist_png/training/0", "./mnist_png/training/1", 
                       "./mnist_png/testing/0", "./mnist_png/testing/1"];
    
    println!("Checking images in dirs {:?}...", directories);

    let out_of_range_images: Vec<String> = vec![];
    let non_8_bit_images: Vec<String> = vec![];


    for dir in &directories {
        for entry in std::fs::read_dir(dir).expect("Failed to read directory") {
            let entry = entry.expect("Failed to read entry");
            let img = image::open(entry.path())
                .expect(format!("Failed to open image {}", entry.path().display()).as_str())
                .to_luma8();
            if !is_within_range(&img) {
                println!("Image {:?} has pixels out of range!", entry.path());
            }
            if !is_8_bit_grayscale(&entry.path()) {
                println!("Image {:?} is not an 8-bit grayscale image!", entry.path());
            }
            // println!("all good for image {:?} ", entry);
        }
    }
}