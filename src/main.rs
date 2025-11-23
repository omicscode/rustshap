use burn::prelude::*;
use burn::tensor::backend::NdArrayBackend;
use ndarray::Array4;
mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod deepplot;
mod deepshap;
mod plot;
mod shap;
use burn::prelude::*;
use burn::tensor::backend::NdArrayBackend;
use deepplot::shapdeepmap;
use deepshap::{DeepShapModel, deepshap_from_background};
use shap::{ShapModel, kernel_shap};

/*
Gaurav Sablok
codeprog@icloud.com
*/

fn main() {
    let argparse = CommandParse::parse();
    match &argparse.command {
        Commands::SHAP {
            modelpath,
            instance,
            instanceshape,
            background,
            backgroundshape,
            nsamplesinput,
            device,
        } => {
            type B = NdArrayBackend<f32>;
            let device = Default::default();
            let modelunwrap = model.load(modelpath).unwrap();
            let backgrounddataset = Array4::from_shape_vec(
                String::from(background).split("").collect::<Vec<_>>(),
                backgroundshape.parse::<usize>(),
            );
            let instance = Array4::<f32>::from_shape_vec(
                String::from(instance).split(" ").collect::<Vec<_>>(),
                instanceshape.parse::<usize>(),
            );
            let shap_vals = kernel_shap(
                &model,
                &instance,
                &background,
                nsamples = nsamplesinput.parse::<usize>(),
                &device,
            );
            println!("SHAP values shape: {:?}", shap_vals.shape());
            println!(
                "Top contributing feature: {:?}",
                shap_vals.iter().enumerate().max_by(|a, b| a
                    .1
                    .abs()
                    .partial_cmp(&b.1.abs())
                    .unwrap())
            );
        }
        Commands::DEEPSHAP {
            modelpath,
            background,
            instanceshape,
            steps,
            position,
        } => {
            type B = NdArrayBackend<f32>;
            type AB = AutodiffBackend<B>;
            let device = Default::default();
            let model = model.load(modelpath).unwrap();
            let backgrounddataset = Array4::from_shape_vec(
                String::from(background).split("").collect::<Vec<_>>(),
                backgroundshape.parse::<usize>(),
            );
            let instance = Array4::<f32>::from_shape_vec(
                String::from(instanceshape).split(" ").collect::<Vec<_>>(),
                instanceshape.parse::<usize>(),
            );
            let shap_values = deepshap_from_background(
                &model,
                &instanceshape,
                &backgrounddataset,
                steps.parse::<usize>().unwrap(),
                &device,
            );
            println!("SHAPvalues: {:?}", shap_values.shape());
            println!("SHAP sum: {:.6}", shap_values.sum());
            shapdeepmap(&shap_values, position, "plotshap.png")?;
        }
    }
}
