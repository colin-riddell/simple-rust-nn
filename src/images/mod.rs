use image::*;

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