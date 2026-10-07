//! Clean a noisy correlation matrix and find its spanning tree. `cargo run --example correlation`
mod common;
use ndarray::Array2;
use quars::math::correlation::{clean_correlation, minimum_spanning_tree};
use quars::portfolio::PortfolioStats;

fn main() -> Result<(), quars::Error> {
    // Six series of 120 observations; the first three share a common factor
    let (n, t) = (6, 120);
    let (factor, noise) = (common::noise(t), common::noise((n + 1) * t));
    let returns = Array2::from_shape_fn((n, t), |(i, k)| {
        noise[(i + 1) * t + k] + if i < 3 { factor[k] } else { 0.0 }
    });
    let assets = (0..n).map(|i| format!("asset {i}")).collect();
    let stats = PortfolioStats::from_returns(assets, returns)?;

    let corr = quars::math::correlation::correlation(&stats.covariance)?;
    println!(
        "sample correlation 0-1 = {:+.3}, 4-5 = {:+.3}",
        corr[[0, 1]],
        corr[[4, 5]]
    );
    let cleaned = clean_correlation(&corr, t)?;
    println!(
        "cleaned correlation 0-1 = {:+.3}, 4-5 = {:+.3}",
        cleaned[[0, 1]],
        cleaned[[4, 5]]
    );
    for (i, j, distance) in minimum_spanning_tree(&corr) {
        println!(
            "tree edge {} - {}  distance {distance:.3}",
            stats.assets[i], stats.assets[j]
        );
    }
    Ok(())
}
