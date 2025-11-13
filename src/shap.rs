use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Float, Tensor};
use ndarray::{Array4, Axis};
use rand::seq::SliceRandom;

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub trait ShapModel<B: Backend> {
    fn predict(&self, x: Tensor<B, 4>) -> Tensor<B, 2>;
}

impl<B: Backend, M: Module<B>> ShapModel<B> for M
where
    M::Record: Send + Sync,
{
    fn predict(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        self.forward(x).sigmoid()
    }
}

pub fn kernel_shap<B: Backend, M: ShapModel<B>>(
    model: &M,
    instance: &Array4<f32>,
    background: &Array4<f32>,
    nsamples: usize,
    device: &B::Device,
) -> Array4<f32> {
    let (n_samples, n_pos, n_feat, n_chan) = background.dim();
    let mut shap_values = Array4::<f32>::zeros((1, n_pos, n_feat, n_chan));
    let mut rng = rand::thread_rng();

    for pos in 0..n_pos {
        for feat in 0..n_feat {
            for chan in 0..n_chan {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for _ in 0..nsamples {
                    let mut mask = vec![false; n_feat * n_chan];
                    let target_idx = feat * n_chan + chan;
                    mask[target_idx] = true;

                    for i in 0..mask.len() {
                        if i != target_idx && rng.gen_bool(0.5) {
                            mask[i] = true;
                        }
                    }

                    let (f_on, f_off) =
                        build_masked_instances(instance, background, pos, &mask, n_feat, n_chan);

                    let x_on = Tensor::from_data(f_on.view().into_data(), device);
                    let x_off = Tensor::from_data(f_off.view().into_data(), device);

                    let p_on = model.predict(x_on).into_scalar();
                    let p_off = model.predict(x_off).into_scalar();
                    let marginal = p_on - p_off;
                    let z = mask.iter().filter(|&&b| b).count();
                    let weight = if z == 0 || z == mask.len() {
                        1e10 // infinite weight approximation
                    } else {
                        (mask.len() - 1) as f32 / (z as f32 * (mask.len() - z) as f32)
                    };

                    weighted_sum += marginal * weight;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    shap_values[[0, pos, feat, chan]] = weighted_sum / total_weight;
                }
            }
        }
    }

    shap_values
}

fn build_masked_instances(
    instance: &Array4<f32>,
    background: &Array4<f32>,
    pos: usize,
    mask: &[bool],
    n_feat: usize,
    n_chan: usize,
) -> (Array4<f32>, Array4<f32>) {
    let mut on = background.mean_axis(Axis(0)).unwrap();
    let mut off = on.clone();

    for f in 0..n_feat {
        for c in 0..n_chan {
            let idx = f * n_chan + c;
            if mask[idx] {
                on[[0, pos, f, c]] = instance[[0, pos, f, c]];
            } else {
                off[[0, pos, f, c]] = instance[[0, pos, f, c]];
            }
        }
    }

    (on.insert_axis(Axis(0)), off.insert_axis(Axis(0)))
}
