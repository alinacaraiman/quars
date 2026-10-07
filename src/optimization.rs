use crate::{
    config::PortofolioOptimization, math::optimization::minimize_quadratic,
    portfolio::PortfolioStats,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::InverseInto;
use std::error::Error;

pub struct Portfolio {
    pub name: &'static str,
    pub weights: Vec<f64>,
    pub expected_return: f64,
    pub std: f64,
}

impl Portfolio {
    fn new(name: &'static str, weights: Array1<f64>, stats: &PortfolioStats) -> Self {
        Self {
            name,
            expected_return: stats.mean_returns.dot(&weights),
            std: weights.dot(&stats.covariance.dot(&weights)).sqrt(),
            weights: weights.to_vec(),
        }
    }
}

/// Every method at the same `params` (τ, then θ defaulting to 0.95), named by its `sub_method`
pub fn optimize_portfolios(
    stats: &PortfolioStats,
    po: &PortofolioOptimization,
) -> Result<Vec<Portfolio>, Box<dyn Error>> {
    let tau = *po.params.first().ok_or("params must start with tau.")?;
    let theta = po.params.get(1).copied().unwrap_or(0.95);
    let (mean, cov) = (&stats.mean_returns, &stats.covariance);
    let n = mean.len();
    let weights = [
        ("equal-weight", Array1::from_elem(n, 1.0 / n as f64)),
        ("mvo", near_optimal_weights(mean, cov, tau, 1.0, false)?),
        (
            "near-optimal",
            near_optimal_weights(mean, cov, tau, theta, false)?,
        ),
        (
            "near-optimal-short",
            near_optimal_weights(mean, cov, tau, theta, true)?,
        ),
        (
            "risk-adjusted",
            risk_adjusted_weights(stats, po.risk_free_rate, tau)?,
        ),
    ];
    Ok(weights
        .into_iter()
        .map(|(name, w)| Portfolio::new(name, w, stats))
        .collect())
}

/// Long-only efficient frontier as (std, expected return), swept over risk aversion
pub fn efficient_frontier(stats: &PortfolioStats, n_points: usize) -> Vec<(f64, f64)> {
    let (mean, cov) = (&stats.mean_returns, &stats.covariance);
    let spread = mean.fold(f64::MIN, |a, &b| a.max(b)) - mean.fold(f64::MAX, |a, &b| a.min(b));
    let scale = spread / cov.diag().fold(f64::MIN, |a, &b| a.max(b));
    let mut x = Array1::from_elem(mean.len(), 1.0 / mean.len() as f64);
    (0..n_points)
        .map(|i| {
            let gamma = scale * 10f64.powf(3.0 - 5.0 * i as f64 / (n_points - 1) as f64);
            x = minimize_quadratic(&(cov * gamma), mean, x.clone(), false);
            (x.dot(&cov.dot(&x)).sqrt(), mean.dot(&x))
        })
        .collect()
}

fn risk_adjusted_weights(
    stats: &PortfolioStats,
    risk_free_rate: f64,
    tau: f64,
) -> Result<Array1<f64>, Box<dyn Error>> {
    let n = stats.assets.len();
    let cov_inv: Array2<f64> = stats.covariance.clone().inv_into()?;
    let ones = Array1::<f64>::ones(n);
    let excess = &stats.mean_returns - annual_to_daily_rate(risk_free_rate);
    let a = ones.dot(&cov_inv.dot(&ones));
    let b = ones.dot(&cov_inv.dot(&excess));
    let lambda_multiplier = (b - 2.0 * tau) / a;
    let optimal_risky = cov_inv.dot(&(&excess - lambda_multiplier)) / (2.0 * tau);
    if (optimal_risky.sum() - 1.0).abs() > 1e-6 {
        return Err("Optimal risky weights do not sum to 1.".into());
    }
    Ok(optimal_risky)
}

/// Near-optimality method (Lolic, 2024).
/// Step 1: maximize utility μᵀx − ½τxᵀΣx, giving ε.
/// Step 2: minimize concentration xᵀx subject to utility ≥ θε.
/// Both subject to 1ᵀx = 1, and x ≥ 0 unless `short`.
fn near_optimal_weights(
    mean: &Array1<f64>,
    cov: &Array2<f64>,
    tau: f64,
    theta: f64,
    short: bool,
) -> Result<Array1<f64>, Box<dyn Error>> {
    let n = mean.len();
    let utility = |x: &Array1<f64>| mean.dot(x) - 0.5 * tau * x.dot(&cov.dot(x));
    let x_equal = Array1::from_elem(n, 1.0 / n as f64);
    let x_mvo = minimize_quadratic(&(cov * tau), mean, x_equal.clone(), short);
    let floor = theta * utility(&x_mvo);
    if utility(&x_equal) >= floor {
        return Ok(x_equal);
    }
    if theta >= 1.0 {
        return Ok(x_mvo);
    }

    // Stationarity of step 2 for a multiplier λ: min ½xᵀ(2I + λτΣ)x − λμᵀx.
    // Utility grows with λ, so bisect λ until it meets the floor.
    let solve = |lambda: f64, x: Array1<f64>| {
        let q = Array2::<f64>::eye(n) * 2.0 + cov * (lambda * tau);
        minimize_quadratic(&q, &(mean * lambda), x, short)
    };
    let (mut lo, mut hi) = (0.0, 1.0);
    let mut x = solve(hi, x_equal);
    while utility(&x) < floor {
        if hi > 1e12 {
            return Err("Utility floor is not reachable.".into());
        }
        lo = hi;
        hi *= 2.0;
        x = solve(hi, x);
    }
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        x = solve(mid, x);
        if utility(&x) < floor {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(solve(hi, x))
}

pub fn annual_to_daily_rate(r_annual: f64) -> f64 {
    (1.0 + r_annual).powf(1.0 / 252.0) - 1.0
}
#[cfg(test)]
mod tests {
    use super::*;

    // Capital markets assumptions of Lolic (2024), Appendix A
    const MEAN: [f64; 10] = [
        0.085, 0.09, 0.1, 0.095, 0.04, 0.055, 0.05, 0.07, 0.075, 0.075,
    ];
    const STD: [f64; 10] = [0.17, 0.19, 0.17, 0.2, 0.0, 0.04, 0.045, 0.11, 0.1, 0.17];
    const CORR: [[f64; 10]; 10] = [
        [1.0, 0.9, 0.7, 0.8, 0.0, 0.0, 0.0, 0.4, 0.5, 0.8],
        [0.9, 1.0, 0.7, 0.75, 0.0, 0.0, 0.0, 0.4, 0.5, 0.85],
        [0.7, 0.7, 1.0, 0.7, 0.0, 0.0, 0.0, 0.5, 0.4, 0.7],
        [0.8, 0.75, 0.7, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.7],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.4, 0.1, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 0.3, 0.1, 0.2],
        [0.4, 0.4, 0.5, 0.4, 0.0, 0.4, 0.3, 1.0, 0.5, 0.4],
        [0.5, 0.5, 0.4, 0.5, 0.0, 0.1, 0.1, 0.5, 1.0, 0.5],
        [0.8, 0.85, 0.7, 0.7, 0.0, 0.2, 0.2, 0.4, 0.5, 1.0],
    ];

    fn weights(theta: f64, short: bool) -> Vec<f64> {
        let mean = Array1::from_vec(MEAN.to_vec());
        let cov = Array2::from_shape_fn((10, 10), |(i, j)| STD[i] * STD[j] * CORR[i][j]);
        near_optimal_weights(&mean, &cov, 3.0, theta, short)
            .unwrap()
            .to_vec()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-3, "{actual:?}");
        }
    }

    #[test]
    fn mvo_matches_paper_table_1() {
        let expected = [0.0, 0.0, 0.438, 0.0, 0.0, 0.158, 0.0, 0.0, 0.404, 0.0];
        assert_close(&weights(1.0, false), &expected);
    }

    #[test]
    fn near_optimal_long_only() {
        let expected = [
            0.0669, 0.0739, 0.2055, 0.0986, 0.0166, 0.147, 0.0985, 0.1145, 0.1786, 0.0,
        ];
        assert_close(&weights(0.95, false), &expected);
    }

    #[test]
    fn near_optimal_short() {
        let expected = [
            0.0573, 0.6462, 1.0363, 0.0414, -4.1688, 3.2621, 1.2487, -0.8191, 1.0511, -1.3552,
        ];
        assert_close(&weights(0.95, true), &expected);
    }

    #[test]
    fn low_floor_gives_equal_weight() {
        assert_close(&weights(0.5, false), &[0.1; 10]);
    }
}
