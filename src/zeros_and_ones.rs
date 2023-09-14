
use rand::*;
use serde::{Serialize, Deserialize};
use std::io::Write;
use std::fs::*;

use crate::utils::random_f64;
use crate::utils::get_files;
use crate::utils::flatten;
use crate::images::load_image;

use crate::network::Network::Network;

pub fn train_on_zeros_and_ones() {
    let mut initial_weights: Vec<f64> = vec![];
    for _ in 0..784 {
        initial_weights.push(random_f64()); // Use a random value instead of 0
    }

    let mut network = Network::new(784, 10, 2);


    // get all files for 1's
    let one_files = get_files("./mnist_png/training/1");
    let number_of_files = one_files.len();

    one_files.iter()
        .enumerate()
        .for_each(|(index, file)| {
            // print_percent_complete(number_of_files, index);
            let inputs = load_image(file);
            // println!("{:?}", inputs);
            network.train(&inputs, &vec![vec![0.0,1.0]], 0.1, 50);
        });


    // get all files for 0's
    let zero_files = get_files("./mnist_png/training/0");

    // loop over all files
    for (index, file) in zero_files.iter().enumerate() {
        // load one into memory
        let pixels = load_image(&file);
        // print_percent_complete(number_of_files, index);

        let outputs = vec![vec![0.0; 1]];

        network.train(&pixels, &vec![vec![1.0, 0.0]], 0.1, 50);
    }


    //copy  trained network to file
    // let mut file = File::create("./network_after_training.txt").unwrap();
    // file.write_all(network.to_string().as_bytes()).unwrap();
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