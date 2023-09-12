use rand::*;
use std::io::Write;
use std::fs::*;
use image::*;


// sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
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

#[derive(Debug)]
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

#[derive(Debug)]
struct Network {
    hidden_layer: Layer,
    output_layer: Layer,
}

impl Network {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Network {
        let hidden_layer = Layer::new(input_size, hidden_size);
        let output_layer = Layer::new(hidden_size, output_size);
        Network { hidden_layer, output_layer }
    }

        //to_string
    // fn to_string(&self) -> String {
    //     let mut s = String::new();
    //     for neuron in self.neurons.iter() {
    //         s.push_str(&format!("{:?}\n", neuron));
    //     }
    //     s
    // }

    fn predict(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let hidden_outputs = self.hidden_layer.predict(inputs);
        self.output_layer.predict(&hidden_outputs)
    }

    // A simple training function with backpropagation omitted for simplicity ?? is this bp ?

    fn train(&mut self, inputs: &Vec<Vec<f64>>, outputs: &Vec<Vec<f64>>, rate: f64, epochs: usize) {
        for _ in 0..epochs {
            for (input, output) in inputs.iter().zip(outputs.iter()) {
                // Forward pass
                let hidden_output = self.hidden_layer.predict(input);
                let mut output_preds = self.output_layer.predict(&hidden_output);
    
                // Compute output layer error
                for (output_pred, target) in output_preds.iter_mut().zip(output.iter()) {
                    *output_pred = *output_pred * (1.0 - *output_pred) * (*target - *output_pred);
                }
    
                // Update output layer weights and biases
                for (neuron, output_pred) in self.output_layer.neurons.iter_mut().zip(output_preds.iter()) {
                    for (weight, hidden_val) in neuron.weights.iter_mut().zip(hidden_output.iter()) {
                        *weight += rate * hidden_val * *output_pred;
                    }
                    neuron.bias += rate * *output_pred;
                }
    
                // Compute hidden layer error and update weights and biases
                // Note: This is a very simplified form and does not follow backpropagation algorithm rigorously
                for (neuron, hidden_val) in self.hidden_layer.neurons.iter_mut().zip(hidden_output.iter()) {
                    let error = hidden_val * (1.0 - hidden_val) * output_preds[0];  // Assuming single output neuron
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

// flatten 2d vec to 1d vec
fn flatten(inputs: &Vec<Vec<f64>>) -> std::vec::Vec<f64> {
    let mut output = Vec::new();
    for row in inputs {
        for col in row {
            output.push(*col);
        }
    }
    return output;
}

// load image file from disk
fn load_image(path: &str) -> Vec<Vec<f64>> {
    let img = image::open(path).unwrap();
    let (width, height) = img.dimensions();
    let mut pixels = Vec::new();
    for y in 0..height {
        let mut row = Vec::new();
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as f64 / 255.0;
            let g = pixel[1] as f64 / 255.0;
            let b = pixel[2] as f64 / 255.0;
            row.push(r);
            row.push(g);
            row.push(b);
        }
        pixels.push(row);
    }
    pixels
}
// get list of files in directory
fn get_files(dir: &str) -> Vec<String> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            files.push(path.to_str().unwrap().to_string());
        }
    }
    files
}
fn train_on_zeros_and_ones() {
    // Create network with one neuron with 28*28 inputs and one output
    let mut initial_weights: Vec<f64> = vec![];
    for _ in 0..784 {
        initial_weights.push(random_f64()); // Use a random value instead of 0
    }

    let mut network = Network::new(784, 10, 2);  // 3 inputs, 2 hidden neurons, 1 output neuron

    // let mut network = Network::new(vec![
    //     Neuron::new(initial_weights, random_f64()), // Randomly initialize the bias as well
    //     // Neuron::new(vec![0.0; 784], 0.0)
    // ]);


    // println!("{:?}", network.neurons);
    //copy network to file
    // let mut file = File::create("./before_training.txt").unwrap();
    // file.write_all(network.to_string().as_bytes()).unwrap();

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

    // println!("{:?}", network.neurons);
    //copy network to file
    // let mut file = File::create("./network_after_training.txt").unwrap();
    // file.write_all(network.to_string().as_bytes()).unwrap();


    
    // load file from  test set
    let zero_test_files = get_files("./mnist_png/testing/0");

    for (index, file) in zero_test_files.iter().enumerate() {
        println!("Predicting file {}", file);
        let pixels = load_image(&file);

        let flattened = flatten(&pixels);

        println!("{:?}",network.predict(&flattened));
    }

    let one_test_files = get_files("./mnist_png/testing/1");

    // for (index, file) in one_test_files.iter().enumerate() {
    //     println!("Predicting file {}", file);
    //     let pixels = load_image(&file);

    //     let flattened = flatten(&pixels);

    //     println!("{:?}",network.predict(&flattened));
    // }
 


    
}

fn train_simple_problem() {
    let mut network = Network::new(3, 2, 1);  // 3 inputs, 2 hidden neurons, 1 output neuron

    let inputs = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    
    let outputs = vec![
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![1.0],
    ];

    network.train(&inputs, &outputs, 0.1, 1000);

    println!("**************** OUTPUT ****************");
    println!("{:?}", network.predict(&vec![1.0, 0.0, 0.0])); // 0
    println!("{:?}", network.predict(&vec![0.0, 1.0, 0.0])); // 1
    println!("{:?}", network.predict(&vec![0.0, 0.0, 1.0])); // 0
    println!("{:?}", network.predict(&vec![1.0, 1.0, 0.0])); // 1
    println!("{:?}", network.predict(&vec![1.0, 0.0, 1.0])); // 0
    println!("{:?}", network.predict(&vec![0.0, 1.0, 1.0])); // 1
    println!("{:?}", network.predict(&vec![1.0, 1.0, 1.0])); // 1
}

fn main() {
    // train_simple_problem();
    train_on_zeros_and_ones();
//     Predicting file ./mnist_png/testing/1/345.png
// [0.00000013757348994826562, 1.0]
// Predicting file ./mnist_png/testing/1/8187.png
// [0.00000013757351324860064, 1.0]
}