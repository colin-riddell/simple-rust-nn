use rand::*;

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

    fn predict(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = Vec::new();
        println!(" iterating over {} neurons", self.neurons.len());
        for neuron in self.neurons.iter() {
     
            outputs.push(neuron.predict(inputs));
        }
        outputs
    }

    fn random_f64(&mut self) -> f64 {
        rand::random::<f64>()
    }
    

    fn train(&mut self, inputs: &Vec<Vec<f64>>, outputs: &Vec<Vec<f64>>, rate: f64) {
        for _ in 0..10000 {
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
        }
    }
}


fn main() {
    // create network with 3 neurons
    let mut network = Network::new(vec![
        Neuron::new(vec![0.0, 0.0, 0.0], 0.0),
        Neuron::new(vec![0.0, 0.0, 0.0], 0.0),
        Neuron::new(vec![0.0, 0.0, 0.0], 0.0),
    ]);


    // train network
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
