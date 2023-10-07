use rand::*;
use rand_distr::{Distribution, Normal};
use std::io::Write;
use std::fs::*;
use image::*;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, Read};


// sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn xavier_initialization(input_size: usize, output_size: usize) -> f64 {
    let var = 2.0 / (input_size + output_size) as f64;
    let stddev = var.sqrt();
    let normal = Normal::new(0.0, stddev).unwrap();  // Note: This can panic if stddev is not positive.
    let mut rng = rand::thread_rng();
    normal.sample(&mut rng)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    delta_weights: Vec<f64>,  // To store momentum
    delta_bias: f64,  // To store momentum
}

const LEARNING_RATE_DECAY: f64 = 0.995;  // Adjust as needed
const MOMENTUM: f64 = 0.9;  // Adjust as needed
const L2_REG: f64 = 0.00001;  // L2 regularization strength

impl Neuron {
    fn new(input_size: usize) -> Neuron {
        // let weights = vec![random_f64(); input_size];
        let weights = (0..input_size).map(|_| xavier_initialization(input_size, 1)).collect::<Vec<_>>();
        let delta_weights = vec![0.0; input_size];
        let bias = random_f64();
        let delta_bias = 0.0;
        Neuron { weights, bias, delta_weights, delta_bias }
    }

    fn predict(&self, inputs: &Vec<f64>) -> f64 {
        let mut sum = 0.0;
        for (weight, input) in self.weights.iter().zip(inputs.iter()) {
            sum += weight * input;
        }
        sum += self.bias;
        sigmoid(sum)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Layer {
        let neurons = vec![Neuron::new(input_size); output_size];
        Layer { neurons }
    }

    fn predict(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.predict(inputs)).collect()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    hidden_layer: Layer,
    output_layer: Layer,
}

impl Network {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Network {
        let hidden_layer = Layer::new(input_size, hidden_size);
        let output_layer = Layer::new(hidden_size, output_size);
        Network { hidden_layer, output_layer }
    }

    pub fn to_string(&self) -> String {
        // Convert the Network instance to a JSON string
        serde_json::to_string(self).expect("Failed to serialize to string")
    }

    pub fn from_string(s: &str) -> Network {
        // Convert a JSON string to a Network instance
        serde_json::from_str(s).expect("Failed to deserialize from string")
    }

    pub fn save(&mut self, filename: &str){
        // save trained neural network to JSON
        let mut file = File::create(filename).unwrap();
        file.write_all(self.to_string().as_bytes()).unwrap();
    }

    pub fn load(filename: &str, ) -> Self {
        // Open the file for reading
        let file = File::open(filename).expect("Failed to open file");

        // Read the file's contents to a String
        let mut content = String::new();
        let mut reader = BufReader::new(file);
        reader.read_to_string(&mut content).expect("Failed to read file content");

        // Deserialize the String to produce a Network instance
        serde_json::from_str(&content).expect("Failed to deserialize network from string")
    }

    pub fn predict(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let hidden_outputs = self.hidden_layer.predict(inputs);
        self.output_layer.predict(&hidden_outputs)
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f64>>, outputs: &Vec<Vec<f64>>, rate: f64, epochs: usize) {
        let mut current_rate = rate;

        for _ in 0..epochs {
            for (input, output) in inputs.iter().zip(outputs.iter()) {
                let hidden_output = self.hidden_layer.predict(input);
                let mut output_preds = self.output_layer.predict(&hidden_output);
                
                // Compute output layer error
                let mut output_errors: Vec<f64> = Vec::new();
                for (output_pred, target) in output_preds.iter().zip(output.iter()) {
                    output_errors.push(*output_pred * (1.0 - *output_pred) * (*target - *output_pred));
                }
                
                  // Update output layer weights and biases
                for (neuron, output_error) in self.output_layer.neurons.iter_mut().zip(output_errors.iter()) {
                    for ((weight, hidden_val), delta_weight) in 
                        neuron.weights.iter_mut().zip(hidden_output.iter()).zip(neuron.delta_weights.iter_mut()) {
                        let weight_update = rate * hidden_val * *output_error + MOMENTUM * *delta_weight;
                        *weight += weight_update - rate * L2_REG * *weight;  // Apply L2 regularization
                        *delta_weight = weight_update;
                    }
                    let bias_update = rate * *output_error + MOMENTUM * neuron.delta_bias;
                    neuron.bias += bias_update;
                    neuron.delta_bias = bias_update;
                }
                for (i, (neuron, hidden_val)) in self.hidden_layer.neurons.iter_mut().zip(hidden_output.iter()).enumerate() {
      
                    let error: f64 = output_errors.iter()
                        .zip(self.output_layer.neurons.iter())
                        .map(|(output_error, output_neuron)| output_neuron.weights[i] * output_error)
                        .sum();


                    for ((weight, input_val), delta_weight) in
                        neuron.weights.iter_mut().zip(input.iter()).zip(neuron.delta_weights.iter_mut()) {
                        let weight_update = rate * input_val * error + MOMENTUM * *delta_weight;
                        *weight += weight_update - rate * L2_REG * *weight;  // Apply L2 regularization
                        *delta_weight = weight_update;
                    }
                    let bias_update = rate * error + MOMENTUM * neuron.delta_bias;
                    neuron.bias += bias_update;
                    neuron.delta_bias = bias_update;
                }

            }
            current_rate *= LEARNING_RATE_DECAY;  // Decay the learning rate

        }
    }
    
}

// random_f64
fn random_f64() -> f64 {
    rand::random::<f64>()
}