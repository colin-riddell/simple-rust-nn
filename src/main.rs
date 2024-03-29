use rand::*;
use serde::{Serialize, Deserialize};

mod network;
mod images;
mod utils;

mod mnist;
mod simple_problem;
mod mnist2;  // The new mnist2 module

use mnist::TrainableCharacterModel;
use mnist::FullMnist;
use simple_problem::train_simple_problem;

fn main() {
    // train_simple_problem();
    let mut mnistModel = FullMnist::new();
    // mnistModel.train();
    // mnistModel.save("./mnist.json");
    mnistModel.load("./mnist.json");
    // mnistModel.test_one("1");
    mnistModel.test_all();
    mnistModel.test_single("./mnist_png/testing/0/3401.png");
    mnistModel.test_single("./mnist_png/testing/1/3852.png");

}