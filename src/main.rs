use burn::prelude::*;
use burn::tensor::backend::NdArrayBackend;
mod plot;
mod shap;
use shap::{ShapModel, kernel_shap};

/*
Gaurav Sablok
codeprog@icloud.com

- adding a clap before the final release.
- adding a function for the background data addition.
- addition a model read directly from the PyTorch also.
*/

fn main() {
    type B = NdArrayBackend<f32>;
    let device = Default::default();
    let model = CNNModel::<B>::new(4, &device);
    let background = Array4::<f32>::zeros((1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
    let instance = Array4::<f32>::from_shape_fn((1, 50, 10, 4), |_| rand::random());
    let shap_vals = kernel_shap(&model, &instance, &background, nsamples = 100, &device);
    println!("SHAP values shape: {:?}", shap_vals.shape());
    println!(
        "Top contributing feature: {:?}",
        shap_vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
    );
}
