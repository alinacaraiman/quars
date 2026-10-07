use crate::data::HistoricalData;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::error::Error;

pub struct PortfolioStats {
    pub assets: Vec<String>,
    pub mean_returns: Array1<f64>,
    pub covariance: Array2<f64>,
    pub returns_matrix: Array2<f64>, // shape: (n_assets, n_samples)
}

pub fn calculate_portfolio_stats(data: &HistoricalData) -> Result<PortfolioStats, Box<dyn Error>> {
    // Group prices by asset
    let mut asset_prices: HashMap<String, Vec<f64>> = HashMap::new();
    for record in data {
        asset_prices
            .entry(record.asset.clone())
            .or_default()
            .push(record.price);
    }

    let mut assets: Vec<String> = asset_prices.keys().cloned().collect();
    assets.sort();
    let n = assets.len();
    if n == 0 {
        return Err("No assets found in data.".into());
    }

    // Find the minimal length of price vector to handle partial data. This is simplified logic.
    let mut min_len = usize::MAX;
    for (_asset, prices) in asset_prices.iter() {
        if prices.len() < min_len {
            min_len = prices.len();
        }
    }
    if min_len < 2 {
        return Err("Not enough data points to compute returns.".into());
    }

    let t = min_len - 1;
    let mut returns_matrix = Array2::<f64>::zeros((n, t));
    for (i, asset) in assets.iter().enumerate() {
        let prices = &asset_prices[asset][0..min_len];
        for day in 0..(min_len - 1) {
            let ret = (prices[day + 1] - prices[day]) / prices[day];
            returns_matrix[[i, day]] = ret;
        }
    }

    let mean_returns = returns_matrix
        .mean_axis(Axis(1))
        .ok_or("Failed to compute mean returns")?;

    // 4. Compute sample covariance
    //    Cov = 1/(T-1) * (R_centered * R_centered^T)
    let covariance = compute_sample_covariance(&returns_matrix)?;

    Ok(PortfolioStats {
        assets,
        mean_returns,
        covariance,
        returns_matrix,
    })
}

/// Compute sample covariance from (n_assets x n_samples) returns
fn compute_sample_covariance(returns: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
    let (n_assets, n_obs) = returns.dim();
    if n_obs < 2 {
        return Err("Not enough observations to compute covariance.".into());
    }

    let means = returns
        .mean_axis(Axis(1))
        .ok_or("Could not compute means of returns matrix")?;

    let mut centered = returns.clone();
    for i in 0..n_assets {
        for j in 0..n_obs {
            centered[[i, j]] -= means[i];
        }
    }

    //  Cov = (1 / (n_obs - 1)) * (centered * centered^T)
    let factor = 1.0 / (n_obs as f64 - 1.0);
    let cov = factor * centered.dot(&centered.t());

    Ok(cov)
}

/// Compute daily portfolio returns from each asset's returns_matrix and weights
pub fn compute_portfolio_returns(returns_matrix: &Array2<f64>, weights: &[f64]) -> Vec<f64> {
    let (n_assets, n_samples) = returns_matrix.dim();
    assert_eq!(
        n_assets,
        weights.len(),
        "Weights length doesn't match assets!"
    );

    let mut port_returns = Vec::with_capacity(n_samples);
    for t in 0..n_samples {
        let mut ret_t = 0.0;
        for i in 0..n_assets {
            ret_t += returns_matrix[[i, t]] * weights[i];
        }
        port_returns.push(ret_t);
    }
    port_returns
}

fn sorted_tail(returns: &[f64], alpha: f64) -> Option<(Vec<f64>, usize)> {
    if returns.is_empty() {
        return None;
    }
    let mut sorted = returns.to_owned();
    sorted.sort_by(f64::total_cmp);
    let idx = ((1.0 - alpha) * sorted.len() as f64).ceil() as usize;
    Some((sorted, idx))
}

/// Historical VaR, `None` if `returns` is empty
pub fn portfolio_var(returns: &[f64], alpha: f64) -> Option<f64> {
    let (sorted, idx) = sorted_tail(returns, alpha)?;
    Some(sorted[idx.min(sorted.len() - 1)])
}

/// Historical CVaR, `None` if `returns` is empty
pub fn portfolio_cvar(returns: &[f64], alpha: f64) -> Option<f64> {
    let (sorted, idx) = sorted_tail(returns, alpha)?;
    let tail = &sorted[..idx.clamp(1, sorted.len())];
    Some(tail.iter().sum::<f64>() / tail.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_cvar() {
        let r: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!(portfolio_var(&r, 0.75), Some(26.0));
        assert_eq!(portfolio_cvar(&r, 0.75), Some(13.0));
    }

    #[test]
    fn empty_and_nan() {
        assert_eq!(portfolio_var(&[], 0.95), None);
        assert_eq!(portfolio_cvar(&[], 0.95), None);
        assert!(portfolio_var(&[f64::NAN, -1.0], 0.95).is_some());
        assert_eq!(portfolio_cvar(&[1.0], 0.99), Some(1.0));
    }
}
