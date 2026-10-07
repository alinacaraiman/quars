//! Price an option on a futures contract and recover its volatility. `cargo run --example options`
use quars::options::{black76, black76_greeks, implied_volatility, OptionKind};

fn main() -> Result<(), quars::Error> {
    // Futures at 105, strike 100, 3% rate, 20% volatility, nine months to expiry
    let (forward, strike, rate, volatility, expiry) = (105.0, 100.0, 0.03, 0.2, 0.75);
    for kind in [OptionKind::Call, OptionKind::Put] {
        let price = black76(kind, forward, strike, rate, volatility, expiry);
        let greeks = black76_greeks(kind, forward, strike, rate, volatility, expiry);
        let implied = implied_volatility(kind, price, forward, strike, rate, expiry)?;
        println!(
            "{kind:?}: price {price:.4}  delta {:+.4}  gamma {:.4}  vega {:.3}  theta {:+.3}  implied vol {implied:.4}",
            greeks.delta, greeks.gamma, greeks.vega, greeks.theta
        );
    }
    Ok(())
}
