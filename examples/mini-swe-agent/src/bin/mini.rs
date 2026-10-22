//! `mini`: mini-swe-agent in the local environment.

use clap::Parser as _;
use delta_mini::cli::{MiniArgs, run_mini};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> std::process::ExitCode {
    run_mini(MiniArgs::parse()).await
}
