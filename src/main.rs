use rand::Rng;

type NetworkParams = (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>);

fn main() {
    let result = init_params();
    let (w1, b1, w2, b2) = result;
    dbg!(w1.len());
    dbg!(b1.len());
    dbg!(w2.len());
    dbg!(b2.len());
}

fn init_params() -> NetworkParams {
    let w1: Vec<Vec<f64>> = generate_random_matrix(10, 784, -0.5, 0.5);
    let b1: Vec<f64> = generate_random_vector(10, -0.5, 0.5);
    let w2: Vec<Vec<f64>> = generate_random_matrix(10, 10, -0.5, 0.5);
    let b2: Vec<f64> = generate_random_vector(10, -0.5, 0.5);

    return (w1, b1, w2, b2);
}

fn generate_random_matrix(rows: usize, cols: usize, min: f64, max: f64) -> Vec<Vec<f64>> {
    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        matrix[i] = generate_random_vector(cols, min, max);
    }

    matrix
}

fn generate_random_vector(cols: usize, min: f64, max: f64) -> Vec<f64> {
    // Initialize a 2D array
    let mut vector: Vec<f64> = vec![0.0; cols];

    // Create a random number generator
    let mut rng = rand::thread_rng();

    // Fill the array with random numbers between MIN and MAX
    for i in 0..cols {
        vector[i] = rng.gen_range(min..=max);
    }

    vector
}

#[allow(dead_code)]
fn relu(z: &Vec<f64>) -> Vec<f64> {
    z.iter().map(|x| x.max(0.0)).collect()
}

#[allow(dead_code)]
fn dot_product(a: Vec<f64>, b: Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

#[allow(dead_code)]
fn softmax(z: &Vec<f64>) -> Vec<f64> {
    let max = z
        .iter()
        .fold(f64::NEG_INFINITY, |prev, curr| prev.max(*curr));
    let exp_sum: f64 = z.iter().map(|x| (x - max).exp()).sum();
    z.iter().map(|x| (x - max).exp() / exp_sum).collect()
}

#[allow(dead_code)]
fn forward_prop(
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    x: Vec<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut z1: Vec<f64> = vec![0.0; w1.len()];
    for i in 0..z1.len() {
        z1[i] = dot_product(w1[i].clone(), x.clone()) + b1[i];
    }
    let a1 = relu(&z1);

    let mut z2: Vec<f64> = vec![0.0; w2.len()];
    for i in 0..z2.len() {
        z2[i] = dot_product(w2[i].clone(), a1.clone()) + b2[i];
    }
    let a2 = softmax(&z2);

    return (z1, a1, z2, a2);
}

#[allow(dead_code)]
fn one_hot_encode(y: Vec<usize>) -> Vec<Vec<usize>> {
    let max: usize = y.iter().fold(0, |prev, curr| prev.max(*curr));
    let mut encoded: Vec<Vec<usize>> = vec![vec![0; max + 1]; y.len()];
    for i in 0..y.len() {
        encoded[i][y[i]] = 1;
    }
    encoded
}

#[allow(dead_code, unused_variables)]
fn back_prop(
    z1: Vec<f64>,
    a1: Vec<f64>,
    z2: Vec<f64>,
    a2: Vec<f64>,
    w2: Vec<Vec<f64>>,
    y: Vec<f64>,
) {
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_size_of_random_matrix() {
        let expected_rows: usize = 10;
        let expected_cols: usize = 13;

        let result = generate_random_matrix(expected_rows, expected_cols, -1.0, 1.0);
        let actual_rows = result.len();
        let actual_cols = result[0].len();
        assert!(
            actual_rows == expected_rows,
            "Expected rows: {:.3}\tActual rows: {:.3}",
            expected_rows,
            actual_rows
        );
        assert!(
            actual_cols == expected_cols,
            "Expected cols: {:.3}\tActual cols: {:.3}",
            expected_cols,
            actual_cols
        );
    }

    #[test]
    fn correct_range_of_random_values() {
        let expected_min: f64 = 0.5;
        let expected_max: f64 = 0.5;
        let result = generate_random_matrix(10, 784, expected_min, expected_max);

        let actual_max: f64 = result
            .iter()
            .map(|f| {
                f.iter()
                    .fold(f64::NEG_INFINITY, |prev, curr| prev.max(*curr))
            })
            .fold(f64::NEG_INFINITY, |prev, curr| prev.max(curr));

        let actual_min: f64 = result
            .iter()
            .map(|f| f.iter().fold(f64::INFINITY, |prev, curr| prev.min(*curr)))
            .fold(f64::INFINITY, |prev, curr| prev.min(curr));

        assert!(
            actual_max <= expected_max,
            "Expected max: {:.3} < Actual max: {:.3}",
            expected_max,
            actual_max
        );
        assert!(
            actual_min >= expected_min,
            "Expected min: {:.3} > Actual min: {:.3}",
            expected_min,
            actual_min
        );
    }

    #[test]
    fn test_relu() {
        let input_vector: Vec<f64> = vec![-1.0, 2.0, -3.0, 4.0, -5.0];
        let expected: Vec<f64> = vec![0.0, 2.0, 0.0, 4.0, 0.0];

        let actual = relu(&input_vector);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_dot_product() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0, 6.0];
        let expected: f64 = 32.0;

        let actual = dot_product(a, b);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_softmax() {
        let input_vector: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0];
        let expected: Vec<f64> = vec![
            0.02364054302159139,
            0.06426165851049616,
            0.17468129859572226,
            0.47483299974438037,
            0.02364054302159139,
            0.06426165851049616,
            0.17468129859572226,
        ];

        let actual = softmax(&input_vector);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_one_hot_encoding() {
        let input_vector: Vec<usize> = vec![1, 2, 0, 1, 2, 0];
        let expected: Vec<Vec<usize>> = vec![
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![1, 0, 0],
        ];

        let actual = one_hot_encode(input_vector);
        assert_eq!(actual, expected);
    }
}
