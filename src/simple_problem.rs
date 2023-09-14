
use crate::network::Network::Network;

pub fn train_simple_problem() {
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