use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "rustshap",
    version = "1.0",
    about = "rustshap.
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// implementation of SHAP for Burn
    SHAP {
        /// path to the model
        modelpath: String,
        /// path to the instance
        instance: String,
        /// path to the instanceshape
        instanceshape: String,
        /// path to the background
        background: String,
        /// path to the background shape
        backgroundshape: String,
        /// number of samples
        nsamplesinput: usize,
        /// device
        device: &B::Device,
    },
    /// implementation of DEEPSHAP for Burn
    DEEPSHAP {
        /// path to the model
        modelpath: String,
        /// path to the background
        background: String,
        /// path to the instance you can supply as a string
        instanceshape: String,
        /// number of steps
        steps: String,
        /// position for the map
        position: usize,
    },
}
