//! Noise against a chaotic logistic map. `cargo run --example nonlinear`
mod common;
use quars::math::nonlinear::{bds, visibility_divergence, zero_one_test};

fn main() -> Result<(), quars::Error> {
    let mut x = 0.3;
    let periodic: Vec<f64> = (0..1000)
        .map(|_| {
            x = 3.55 * x * (1.0 - x);
            x
        })
        .collect();
    let chaotic: Vec<f64> = (0..1000)
        .map(|_| {
            x = 3.97 * x * (1.0 - x);
            x
        })
        .collect();
    for (name, series) in [
        ("noise", common::noise(1000)),
        ("periodic map", periodic),
        ("chaotic map", chaotic),
    ] {
        println!(
            "{name:<13} BDS p = {:.4}  0-1 test = {:+.3}  visibility divergence = {:.4}",
            bds(&series, 2, 1.5)?.p,
            zero_one_test(&series)?,
            visibility_divergence(&series)?
        );
    }
    Ok(())
}
