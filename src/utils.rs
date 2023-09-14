// random_f64
pub fn random_f64() -> f64 {
    rand::random::<f64>()
}

// flatten 2d vec to 1d vec
pub fn flatten(inputs: &Vec<Vec<f64>>) -> std::vec::Vec<f64> {
    let mut output = Vec::new();
    for row in inputs {
        for col in row {
            output.push(*col);
        }
    }
    return output;
}



// get list of files in directory
pub fn get_files(dir: &str) -> Vec<String> {
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