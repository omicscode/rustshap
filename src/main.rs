use burn::prelude::*;
use burn::tensor::backend::NdArrayBackend;
use ndarray::Array4;
mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod plot;
mod shap;
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
    }
}
