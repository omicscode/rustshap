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
    SHAP {
        modelpath: String,
        instance: String,
        instanceshape: String,
        background: String,
        backgroundshape: String,
        nsamplesinput: usize,
        device: &B::Device,
    },
}
