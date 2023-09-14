
use rand::*;
use serde::{Serialize, Deserialize};
use std::io::Write;
use std::fs::*;

use crate::utils::random_f64;
use crate::utils::get_files;
use crate::utils::flatten;
use crate::images::load_image;

use crate::network::Network::Network;

fn get_mnist_letter_training(letter: &str)-> Vec<String>{
    let path = format!("./mnist_png/training/{}", letter);
    return get_files(&path)
}

fn train_letter(network: &mut Network, letter: &str, output: &Vec<f64>) {
    // get all files for 1's
    let one_files = get_mnist_letter_training(letter);

    one_files.iter()
        .enumerate()
        .for_each(|(index, file)| {
            let inputs = load_image(file);
            network.train(&inputs, &vec![output.to_vec()], 0.1, 50);
        });

}

pub fn train_on_zeros_and_ones() {
    let mut initial_weights: Vec<f64> = vec![];
    for _ in 0..784 {
        initial_weights.push(random_f64()); // Use a random value instead of 0
    }

    let mut network = Network::new(784, 10, 2);

    let one_output = vec![0.0, 1.0];
    train_letter(&mut network, "1", &one_output);

    let zero_output = vec![1.0, 0.0];
    train_letter(&mut network, "0", &zero_output);

    // save trained neural network to JSON
    let mut file = File::create("./network_after_training.json").unwrap();
    file.write_all(network.to_string().as_bytes()).unwrap();

    
    // load file from  test set
    let zero_test_files = get_files("./mnist_png/testing/0");

    for (index, file) in zero_test_files.iter().enumerate() {
        println!("Predicting file {}", file);
        let pixels = load_image(&file);

        let flattened = flatten(&pixels);

        println!("{:?}",network.predict(&flattened));
    }

    let one_test_files = get_files("./mnist_png/testing/1");

}

pub fn test_all(){

}