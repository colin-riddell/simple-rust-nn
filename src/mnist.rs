
use image::GrayImage;
use rand::*;
use serde::{Serialize, Deserialize};
use std::io::Write;
use std::fs;

use crate::utils::random_f64;
use crate::utils::get_files;
use crate::utils::flatten;
use crate::images::load_image;
use crate::images::check_image;


use crate::network::Network::Network;

fn vectorized_result(label: usize) -> Vec<f64> {
    let mut vec = vec![0.0; 2];
    vec[label] = 1.0;
    vec
}

fn image_to_vector(img: &GrayImage) -> Vec<f64> {
    img.pixels().map(|p| p[0] as f64 / 255.0).collect()
}



pub trait TrainableCharacterModel {
    fn train(&mut self);
    fn save(&mut self, filename: &str);
    fn load(&mut self, filename: &str);
    fn test_one(&mut self, letter: &str);
    fn test_single(&mut self, location: &str);
    fn test_all(&mut self);
}

// #[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullMnist {
    network: Network
}
impl FullMnist {
    pub fn new() -> FullMnist {

        const IMG_WIDTH: usize = 28;
        const IMG_HEIGHT: usize = 28;

        let mut network = Network::new(IMG_WIDTH * IMG_HEIGHT, 10, 2);

        FullMnist { network }
    }  
}
  


impl TrainableCharacterModel for FullMnist {

    fn train(&mut self) {
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
                else if i == 1{
                    count_ones += 1;
                }
            }
        }
        
        println!("{} {}", train_data.len(), train_labels.len() );
        if(train_data.len() != train_labels.len()){
            println!("Training data and labels not same amount!");
        }

        println!("Training on {} 0's and {} 1's", count_zeros, count_ones);

        self.network.train(&train_data, &train_labels, 0.2, 50);  // Learning rate 0.5 and 30 epochs as an example

    }

    fn save(&mut self, filename: &str ){
        self.network.save(filename);
    }

    fn load(&mut self, filename: &str){
        let loaded_network = Network::load(filename);
        self.network = loaded_network;
    }

    fn test_one(&mut self, letter: &str){

            // load file from  test set
            let zero_test_files = get_mnist_letter_testing(letter);

            for (index, file) in zero_test_files.iter().enumerate() {
                println!("Predicting file {}", file);
                let pixels = load_image(&file);
        
                let flattened = flatten(&pixels);
        
                let predicted_output: Vec<f64> =  self.network.predict(&flattened);
                println!("{:?}", predicted_output);
                println!("{}", decode_letter(&predicted_output).expect("should have only one prominent value in predicted_output vector."));
            }

    }
    

    fn test_all(&mut self) {
        let mut correct_predictions = 0;
    
        for i in 0..=1 {
            let path = format!("./mnist_png/testing/{}", i);
            for entry in fs::read_dir(&path).expect("Failed to read directory") {
                let entry = entry.expect("Failed to read entry");
                let img = image::open(entry.path()).expect("Failed to open image").to_luma8();
                let output = self.network.predict(&image_to_vector(&img));
                println!("Predicting image {:?} as {:?}", entry.path(), output);
    
                if output[0] > output[1] && i == 0 {
                    correct_predictions += 1;
                }
                else if output[0] < output[1] && i == 1 {
                    correct_predictions += 1;
                }
            }
        }
    
        let total_tests_0 = fs::read_dir("./mnist_png/testing/0").unwrap().count();
        let total_tests_1 = fs::read_dir("./mnist_png/testing/1").unwrap().count();
        let total_tests = total_tests_0 + total_tests_1;
    
        let accuracy = (correct_predictions as f64) / (total_tests as f64);
        println!("Accuracy: {}", accuracy);
    }

    fn test_single(&mut self, location: &str) {
        println!("Predicting file {}", location);
        let pixels = load_image(location);

        let flattened = flatten(&pixels);

        let predicted_output: Vec<f64> =  self.network.predict(&flattened);
        println!("{:?}", predicted_output);
        println!("{}", decode_letter(&predicted_output).expect("should have only one prominent value in predicted_output vector."));
    }

}

fn decode_letter(vec: &Vec<f64>) -> Option<String> {
    // Define the full range of characters from '0' to 'z'
    // let full_range = "0123456789abcdefghijklmnopqrstuvwxyz";
    let full_range = "01";

    // Check for one and only one entry set to 1.0
    let ones_count = vec.iter().filter(|&&x| x >= 0.8).count();
    if ones_count != 1 {
        return None;
    }

    // Find the index with the value of 1.0
    if let Some(index) = vec.iter().position(|&x| x >= 0.8) {
        // Check if the index is within the length of full_range
        if index < full_range.len() {
            Some(full_range.chars().nth(index).unwrap().to_string())
        } else {
            // Index out of bounds of the full_range
            None
        }
    } else {
        // No index found with value >0.8
        None
    }
}

fn encode_letter(c: &str) -> Option<Vec<f64>> {
    // Define the full range of characters from '0' to 'z'
    // let full_range = "0123456789abcdefghijklmnopqrstuvwxyz";
    let full_range = "01";

    // Initialize a vector with zeros
    let mut vec = vec![0.0; full_range.len()];
    
    // Find the index of the character
    if let Some(index) = full_range.find(c) {
        // Set the corresponding index to 1.0
        vec[index] = 1.0;
        Some(vec)
    } else {
        // Return None if the character is not in the range
        None
    }
}

fn get_mnist_letter_testing(letter: &str)-> Vec<String>{
    let path = format!("./mnist_png/testing/{}", letter);
    return get_files(&path)
}

fn get_mnist_letter_training(letter: &str)-> Vec<String>{
    let path = format!("./mnist_png/training/{}", letter);
    return get_files(&path)
}
