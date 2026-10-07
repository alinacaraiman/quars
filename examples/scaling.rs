//! Scaling exponents of noise and of a random walk. `cargo run --example scaling`
mod common;
use quars::math::scaling::{dfa, hurst_rs};

fn main() -> Result<(), quars::Error> {
    let scales = [16, 32, 64, 128, 256];
    let noise = common::noise(4096);
    let walk = common::ar1(1.0, 4096);
    println!(
        "noise        DFA = {:.3}  Hurst R/S = {:.3}",
        dfa(&noise, &scales)?,
        hurst_rs(&noise, &scales)?
    );
    println!(
        "random walk  DFA = {:.3}  Hurst R/S = {:.3}",
        dfa(&walk, &scales)?,
        hurst_rs(&walk, &scales)?
    );
    Ok(())
}
