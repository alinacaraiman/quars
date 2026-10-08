//! Compare the portfolio methods on simulated returns. `cargo run --example optimization`
mod common;
use ndarray::Array2;
use quars::optimization::optimize_portfolios;
use quars::portfolio::{compute_portfolio_returns, portfolio_cvar, portfolio_var, PortfolioStats};

fn main() -> Result<(), quars::Error> {
    // Four assets with rising drift and volatility over 500 days
    let (n, t) = (4, 500);
    let noise = common::noise(n * t);
    let returns = Array2::from_shape_fn((n, t), |(i, k)| {
        let scale = 0.01 * (i + 1) as f64;
        0.03 * scale + 2.0 * scale * noise[i * t + k]
    });
    let assets = ["A", "B", "C", "D"].map(String::from).to_vec();
    let stats = PortfolioStats::from_returns(assets, returns)?;

    for p in optimize_portfolios(&stats, 0.0, 3.0, 0.95, 2)? {
        let returns = compute_portfolio_returns(&stats.returns_matrix, &p.weights);
        println!(
            "{:<19} weights {:.2?}  VaR95 {:+.4}  CVaR95 {:+.4}",
            p.name,
            p.weights,
            portfolio_var(&returns, 0.95).unwrap_or(f64::NAN),
            portfolio_cvar(&returns, 0.95).unwrap_or(f64::NAN)
        );
    }
    Ok(())
}
