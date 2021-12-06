use rand::*;
use std::fs::*;
use image::*;

// sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    // new
    fn new(weights: Vec<f64>, bias: f64) -> Neuron {
        Neuron { weights, bias }
    }

    // predict
    fn predict(&self, inputs: &Vec<f64>) -> f64 {
        let mut sum = 0.0;
        for (weight, input) in self.weights.iter().zip(inputs.iter()) {
            sum += weight * input;
        }
        sum += self.bias;
        sigmoid(sum)
    }
}


struct Network {
    neurons: Vec<Neuron>,
}


impl Network {
    fn new(neurons: Vec<Neuron>) -> Network {
        Network { neurons }
    }



    // predict 2d
    fn predict_2d(&self, inputs: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut outputs = Vec::new();
        for inputs in inputs.iter() {
            let mut sum = 0.0;
            for (neuron, input) in self.neurons.iter().zip(inputs.iter()) {
                sum += neuron.predict(&vec![*input]);
            }
            outputs.push(sigmoid(sum));
        }
        outputs
    }

    fn predict(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = Vec::new();
        for neuron in self.neurons.iter() {
            outputs.push(neuron.predict(inputs));
        }
        outputs
    }

    fn random_f64(&mut self) -> f64 {
        rand::random::<f64>()
    }

    // fn new_predict(&mut self, inputs: &Vec<Vec<f64>>, outputs: &Vec<Vec<f64>>, rate: f64){
    //     let mut outputs = Vec::new();

    //     for (input, output) in inputs.iter().zip(outputs.iter()) {
    //         for neuron in self.neurons.iter() {
    //             let mut out = neuron.predict(input);
    //             outputs.push(out);
    //         }
    //     }

    // }

    fn train(&mut self, inputs: &Vec<Vec<f64>>, outputs: &Vec<Vec<f64>>, rate: f64) {
        // for _ in 0..10000 {
            for (input, output) in inputs.iter().zip(outputs.iter()) {
                let mut outputs = self.predict(input);
                for (output, target) in outputs.iter_mut().zip(output.iter()) {
                    *output = *output * (1.0 - *output) * (*target - *output);
                }
                for (neuron, output) in self.neurons.iter_mut().zip(outputs.iter()) {
                    for (weight, input) in neuron.weights.iter_mut().zip(input.iter()) {
                        *weight += rate * *input * *output;
                    }
                    neuron.bias += rate * *output;
                }
            }
        // }
    }
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

    // flatten 2d vec to 1d vec
    fn flatten(inputs: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut output = Vec::new();
        for row in inputs {
            for col in row {
                output.push(*col);
            }
        }
        output
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

fn print_percent_complete(number_of_files: usize, index: usize){
    if index > 0 {
        let percent_complete = index as f32  / number_of_files as f32 * 100.0;
        println!("{:.2}% complete", percent_complete);
    }
}


fn train_on_zeros_and_ones() {

    // create network with one neuron with 28*28 inputs and one output
    let mut network = Network::new(vec![
        Neuron::new(vec![0.0; 784], 0.0),
        // Neuron::new(vec![0.0; 784], 0.0)
    ]);

    // get all files for 1's
    let one_files = get_files("./mnist_png/training/1");
    let number_of_files = one_files.len();

    // loop over all files
    for (index, file) in one_files.iter().enumerate() {
         // load one into memory
        let pixels = load_image(&file);
        print_percent_complete(number_of_files, index);

        network.train(&pixels, &vec![vec![1.0]], 0.1);
    }

    // get all files for 0's
    let zero_files = get_files("./mnist_png/training/0");

    // loop over all files
    for (index, file) in zero_files.iter().enumerate() {
        // load one into memory
        let pixels = load_image(&file);
        print_percent_complete(number_of_files, index);

        network.train(&pixels, &vec![vec![0.0]], 0.1);
    }
    
    // load file from  test set
    let zero_test_files = get_files("./mnist_png/testing/0");

    for (index, file) in zero_test_files.iter().enumerate() {
        println!("Predicting file {}", file);
        let pixels = load_image(&file);

        let flattened = flatten(&pixels);

        println!("{:?}",network.predict(&flattened)); // should be a 0 ish so 0.99, 0.01
    }

    let one_test_files = get_files("./mnist_png/testing/1");

    for (index, file) in one_test_files.iter().enumerate() {
        println!("Predicting file {}", file);
        let pixels = load_image(&file);

        let flattened = flatten(&pixels);

        println!("{:?}",network.predict(&flattened)); // should be a 0 ish so 0.99, 0.01
    }
 


    
}

fn just_one_file(){
        let zero_files = get_files("./mnist_png/training/0");
    let just_one = zero_files[0].clone();
    println!("{} size of one image", just_one.len());
    let pixels = load_image(&just_one);
}

fn train_simple_problem(){
 let mut network = Network::new(vec![
        Neuron::new(vec![0.0, 0.0, 0.0], 0.0),
    ]);

    network.train(&vec![vec![0.0, 0.0, 0.0]], &vec![vec![0.0]], 0.1);
    network.train(&vec![vec![1.0, 0.0, 0.0]], &vec![vec![0.0]], 0.1);
    network.train(&vec![vec![0.0, 1.0, 0.0]], &vec![vec![1.0]], 0.1);
    network.train(&vec![vec![0.0, 0.0, 1.0]], &vec![vec![0.0]], 0.1);
    network.train(&vec![vec![1.0, 1.0, 0.0]], &vec![vec![1.0]], 0.1);
    network.train(&vec![vec![1.0, 0.0, 1.0]], &vec![vec![0.0]], 0.1);
    network.train(&vec![vec![0.0, 1.0, 1.0]], &vec![vec![1.0]], 0.1);
    network.train(&vec![vec![1.0, 1.0, 1.0]], &vec![vec![1.0]], 0.1);

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
    train_on_zeros_and_ones();
    // train_simple_problem();
   
   




}
