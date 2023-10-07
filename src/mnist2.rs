use image::GrayImage;
use crate::network::Network::Network;
use std::fs;
use crate::images::check_image;


const IMG_WIDTH: usize = 28;
const IMG_HEIGHT: usize = 28;

pub fn train_and_test() {
    check_image();
    
    let mut train_data = vec![];
    let mut train_labels = vec![];

    let mut count_zeros: i32 = 0;
    let mut count_ones: i32  = 0;

    for i in 0..=1 {

        let path = format!("./mnist_png/training/{}", i);
        for entry in fs::read_dir(path).expect("Failed to read directory") {
            let entry = entry.expect("Failed to read entry");
            let img = image::open(entry.path()).expect("Failed to open image").to_luma8();
            train_data.push(image_to_vector(&img));
            train_labels.push(vectorized_result(i));
            if i == 0{
                count_zeros += 1;
            }
            if i == 1{
                count_ones += 1;
            }
        }
    }
    
    println!("{} {}", train_data.len(), train_labels.len() );
    if(train_data.len() != train_labels.len()){
        println!("Training data and labels not same amount!");
        // false
    }

    println!("Training on {} 0's and {} 1's", count_zeros, count_ones);

    // println!("Training with train labels {:?}", train_labels);
    let mut network = Network::new(IMG_WIDTH * IMG_HEIGHT, 10, 2);  // 30 hidden neurons as an example
    network.train(&train_data, &train_labels, 0.2, 50);  // Learning rate 0.5 and 30 epochs as an example
    network.save("./mnist2.json");

    // Now testing
    let mut correct_predictions = 0;

    for i in 0..=0 {
        let path = format!("./mnist_png/testing/{}", i);
        for entry in fs::read_dir(path).expect("Failed to read directory") {
            let entry = entry.expect("Failed to read entry");
            let img = image::open(entry.path()).expect("Failed to open image").to_luma8();
            let output = network.predict(&image_to_vector(&img));
            println!("Predicting image {:?} as {:?}", entry.path(), output);

            if output[0] > output[1] && i == 0 {
                correct_predictions += 1;
            }
            if output[0] < output[1] && i == 1 {
                correct_predictions += 1;
            }
        }
    }

    let total_tests = 2 * fs::read_dir("./mnist_png/testing/0").unwrap().count(); // Assuming equal number of 0s and 1s.
    let accuracy = (correct_predictions as f64) / (total_tests as f64);
    println!("Accuracy: {}", accuracy);
}

fn image_to_vector(img: &GrayImage) -> Vec<f64> {
    img.pixels().map(|p| p[0] as f64 / 255.0).collect()
}

fn vectorized_result(label: usize) -> Vec<f64> {
    let mut vec = vec![0.0; 2];
    vec[label] = 1.0;
    vec
}
