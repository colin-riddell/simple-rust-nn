use rand::*;
use serde::{Serialize, Deserialize};

mod network;
mod images;
mod utils;

mod mnist;
mod simple_problem;

use mnist::TrainableCharacterModel;
use mnist::FullMnist;
// use simple_problem::train_simple_problem;

fn main() {
    // train_simple_problem();
    let mut mnistModel = FullMnist::new();
    mnistModel.train();
    // mnistModel.save();
}