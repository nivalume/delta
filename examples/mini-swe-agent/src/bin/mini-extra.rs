//! `mini-extra`: the global config, the trajectory inspector and the SWE-bench runners.

use clap::Parser as _;
use delta_mini::cli::{ExtraArgs, run_extra};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> std::process::ExitCode {
    run_extra(ExtraArgs::parse()).await
}
