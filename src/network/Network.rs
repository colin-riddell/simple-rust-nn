use rand::*;
use std::io::Write;
use std::fs::*;
use image::*;
use serde::{Serialize, Deserialize};

// sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(input_size: usize) -> Neuron {
        let weights = vec![random_f64(); input_size];
        let bias = random_f64();
        Neuron { weights, bias }
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

    pub fn predict(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let hidden_outputs = self.hidden_layer.predict(inputs);
        self.output_layer.predict(&hidden_outputs)
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f64>>, outputs: &Vec<Vec<f64>>, rate: f64, epochs: usize) {
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
                    for (weight, hidden_val) in neuron.weights.iter_mut().zip(hidden_output.iter()) {
                        *weight += rate * hidden_val * *output_error;
                    }
                    neuron.bias += rate * *output_error;
                }
                
    
                // Compute hidden layer error and update weights and biases
                for (neuron, hidden_val) in self.hidden_layer.neurons.iter_mut().zip(hidden_output.iter()) {
                    // For each hidden neuron, sum up the contributions from all output neurons
                    let mut error = 0.0;
                    for (output_error, output_neuron) in output_errors.iter().zip(self.output_layer.neurons.iter()) {
                        error += output_neuron.weights[0] * output_error;  // Assumes uniform length of neuron.weights
                    }
                    error *= hidden_val * (1.0 - hidden_val);

                    // Update the weights and bias for the hidden layer neuron
                    for (weight, input_val) in neuron.weights.iter_mut().zip(input.iter()) {
                        *weight += rate * input_val * error;
                    }
                    neuron.bias += rate * error;
                }
            }
        }
    }
    
}

// random_f64
fn random_f64() -> f64 {
    rand::random::<f64>()
}