
use rand::*;
use serde::{Serialize, Deserialize};
use std::io::Write;
use std::fs::*;

use crate::utils::random_f64;
use crate::utils::get_files;
use crate::utils::flatten;
use crate::images::load_image;

use crate::network::Network::Network;

pub trait TrainableCharacterModel {
    fn train(&mut self);
    fn train_letter(&mut self, letter: &str);
    fn save(&mut self);
    // fn test_all();
    // fn test_one(letter: &str);
    // fn test_single(path: &str);
}

// #[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullMnist {
    network: Network
}
impl FullMnist {
    pub fn new() -> FullMnist {
        let network = Network::new(784, 10, 36);
        FullMnist { network }
    }  
}
  


impl TrainableCharacterModel for FullMnist {


    fn train_letter(&mut self, letter: &str) {
        // get all files for 1's
        let one_files = get_mnist_letter_training(letter);

        if let Some(output) = encode_letter(letter) {
            println!("generating for {} output ", output.len());
            one_files.iter()
                .enumerate()
                .for_each(|(index, file)| {
                    let inputs = load_image(file);
                    self.network.train(&inputs, &vec![output.to_vec()], 0.1, 50);
                });
        }



    }

    fn train(&mut self) {
        let mut initial_weights: Vec<f64> = vec![];
        for _ in 0..784 {
            initial_weights.push(random_f64()); // Use a random value instead of 0
        }


        // let one_output = vec![0.0, 1.0];
        self.train_letter("1");

        // let zero_output = vec![1.0, 0.0];
        self.train_letter("0");


        // let one_test_files = get_files("./mnist_png/testing/1");

    }

    fn save(&mut self){
        // save trained neural network to JSON
        let mut file = File::create("./mnist.json").unwrap();
        file.write_all(self.network.to_string().as_bytes()).unwrap();
    }

    // fn test_all_letter(&self, letter: &str){

    //         // load file from  test set
    //         let zero_test_files = get_mnist_letter_testing("0");

    //         for (index, file) in zero_test_files.iter().enumerate() {
    //             println!("Predicting file {}", file);
    //             let pixels = load_image(&file);
        
    //             let flattened = flatten(&pixels);
        
    //             println!("{:?}",self.network.predict(&flattened));
    //         }

    // }

}

fn encode_letter(c: &str) -> Option<Vec<f64>> {
    // Define the full range of characters from '0' to 'z'
    let full_range = "0123456789abcdefghijklmnopqrstuvwxyz";
    
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
