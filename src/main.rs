use rand::*;
use serde::{Serialize, Deserialize};

mod network;
mod images;
mod utils;

mod zeros_and_ones;
mod simple_problem;

use zeros_and_ones::train_on_zeros_and_ones;
use simple_problem::train_simple_problem;

fn main() {
    // train_simple_problem();
    train_on_zeros_and_ones();
}