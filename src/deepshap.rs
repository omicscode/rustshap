use burn::module::Module;
use burn::tensor::TensorData;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Device, Float, Int, Tensor};
use ndarray::{Array4, Axis};
use std::ops::Add;

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub trait DeepShapModel<B: Backend> {
    fn forward_with_grad(&self, x: Tensor<B, 4>) -> Tensor<B, 2>;
}

impl<B: Backend, M: Module<AutodiffBackend<B>>> DeepShapModel<AutodiffBackend<B>> for M
where
    M::Record: Send + Sync,
{
    fn forward_with_grad(&self, x: Tensor<AutodiffBackend<B>, 4>) -> Tensor<AutodiffBackend<B>, 2> {
        self.forward(x).sigmoid()
    }
}

pub fn deepshap<B: Backend>(
    model: &impl DeepShapModel<AutodiffBackend<B>>,
    instance: &Array4<f32>,
    baseline: &Array4<f32>,
    n_steps: usize,
    device: &B::Device,
) -> Array4<f32>
where
    B: burn::tensor::backend::NdArrayBackend,
{
    let device = device.clone();
    let instance_tensor = Tensor::<B, 4>::from_data(instance.view().into_data(), &device);
    let baseline_tensor = Tensor::<B, 4>::from_data(baseline.view().into_data(), &device);
    let mut shap_values = Array4::<f32>::zeros(instance.dim());
    let model_ad = burn::module::AutodiffModule::from_module(model, &device);

    for step in 1..=n_steps {
        let alpha = step as f32 / n_steps as f32;
        let interpolated = baseline_tensor
            .clone()
            .add(alpha)
            .mul(&instance_tensor.clone());
        let output = model_ad.forward_with_grad(interpolated.requires_grad());
        let grad = output.backward();
        let input_grad = grad.get(&output.input_id()).unwrap();
        let grad_array: Array4<f32> = input_grad
            .into_data()
            .convert::<f32>()
            .value
            .into_dimensionality()
            .unwrap();
        let diff = instance - baseline;
        let contrib = grad_array * &diff;
        shap_values = shap_values + contrib / n_steps as f32;
    }

    shap_values
}

pub fn deepshap_from_background<B: Backend>(
    model: &impl DeepShapModel<AutodiffBackend<B>>,
    instance: &Array4<f32>,
    background: &Array4<f32>,
    n_steps: usize,
    device: &B::Device,
) -> Array4<f32> {
    let baseline = background.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
    deepshap(model, instance, &baseline, n_steps, device)
}
