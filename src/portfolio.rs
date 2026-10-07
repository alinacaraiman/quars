#[cfg(feature = "data")]
use crate::data::HistoricalData;
use crate::Error;
use ndarray::{Array1, Array2, Axis};
#[cfg(feature = "data")]
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub struct PortfolioStats {
    pub assets: Vec<String>,
    pub mean_returns: Array1<f64>,
    pub covariance: Array2<f64>,
    pub returns_matrix: Array2<f64>, // shape: (n_assets, n_samples)
}

impl PortfolioStats {
    /// From an (n_assets, n_samples) returns matrix
    pub fn from_returns(assets: Vec<String>, returns_matrix: Array2<f64>) -> Result<Self, Error> {
        let mean_returns = returns_matrix
            .mean_axis(Axis(1))
            .ok_or("Failed to compute mean returns")?;
        let covariance = compute_sample_covariance(&returns_matrix)?;
        Ok(Self {
            assets,
            mean_returns,
            covariance,
            returns_matrix,
        })
    }
}

#[cfg(feature = "data")]
pub fn calculate_portfolio_stats(data: &HistoricalData) -> Result<PortfolioStats, Error> {
    // Prices by date in chronological order, whatever order the records come in
    let mut by_date: BTreeMap<&str, HashMap<&str, f64>> = BTreeMap::new();
    for record in data {
        by_date
            .entry(&record.date)
            .or_default()
            .insert(&record.asset, record.price);
    }
    let assets: Vec<String> = data
        .iter()
        .map(|record| record.asset.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if assets.is_empty() {
        return Err("No assets found in data.".into());
    }

    // Only dates on which every asset has a price
    let prices: Vec<Vec<f64>> = by_date
        .values()
        .filter_map(|day| {
            assets
                .iter()
                .map(|a| day.get(a.as_str()).copied())
                .collect()
        })
        .collect();
    if prices.len() < 2 {
        return Err("Not enough common dates to compute returns.".into());
    }

    let returns_matrix = Array2::from_shape_fn((assets.len(), prices.len() - 1), |(i, t)| {
        (prices[t + 1][i] - prices[t][i]) / prices[t][i]
    });
    PortfolioStats::from_returns(assets, returns_matrix)
}

/// Compute sample covariance from (n_assets x n_samples) returns
fn compute_sample_covariance(returns: &Array2<f64>) -> Result<Array2<f64>, Error> {
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

    #[cfg(feature = "data")]
    #[test]
    fn stats_align_dates() {
        use crate::data::Record;
        let record = |date: &str, asset: &str, price| Record {
            date: date.into(),
            asset: asset.into(),
            price,
        };
        // Newest first, and B has no price on the 2nd
        let data = vec![
            record("2024-01-04", "A", 121.0),
            record("2024-01-03", "A", 110.0),
            record("2024-01-02", "A", 105.0),
            record("2024-01-01", "A", 100.0),
            record("2024-01-04", "B", 55.0),
            record("2024-01-03", "B", 50.0),
            record("2024-01-01", "B", 40.0),
        ];
        let stats = calculate_portfolio_stats(&data).unwrap();
        let expected = ndarray::array![[0.10, 0.10], [0.25, 0.10]];
        assert!((&stats.returns_matrix - &expected)
            .iter()
            .all(|d| d.abs() < 1e-12));
    }

    #[test]
    fn empty_and_nan() {
        assert_eq!(portfolio_var(&[], 0.95), None);
        assert_eq!(portfolio_cvar(&[], 0.95), None);
        assert!(portfolio_var(&[f64::NAN, -1.0], 0.95).is_some());
        assert_eq!(portfolio_cvar(&[1.0], 0.99), Some(1.0));
    }
}
