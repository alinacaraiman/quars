use crate::Error;
use crate::{math::optimization::minimize_quadratic, portfolio::PortfolioStats};
use ndarray::{Array1, Array2};
#[cfg(feature = "openblas")]
use ndarray_linalg::InverseInto;

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

/// Every method at the same τ, θ and subset size `m`, named by its `sub_method`.
/// `mvo-cleaned` is `mvo` on the random-matrix cleaned covariance, left out when an
/// asset has no variance.
pub fn optimize_portfolios(
    stats: &PortfolioStats,
    risk_free_rate: f64,
    tau: f64,
    theta: f64,
    m: usize,
) -> Result<Vec<Portfolio>, Error> {
    let (mean, cov) = (&stats.mean_returns, &stats.covariance);
    let n = mean.len();
    let mut weights = vec![
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
            risk_adjusted_weights(stats, risk_free_rate, tau)?,
        ),
        ("resampled", resampled_weights(mean, cov, tau, m, 1000)?),
    ];
    if let Ok(cleaned) = stats.cleaned_covariance() {
        weights.push((
            "mvo-cleaned",
            near_optimal_weights(mean, &cleaned, tau, 1.0, false)?,
        ));
    }
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

/// Maximize (μ − r_f)ᵀx − τxᵀΣx subject to 1ᵀx = 1
pub fn risk_adjusted_weights(
    stats: &PortfolioStats,
    risk_free_rate: f64,
    tau: f64,
) -> Result<Array1<f64>, Error> {
    let n = stats.assets.len();
    let excess = &stats.mean_returns - annual_to_daily_rate(risk_free_rate);
    let optimal_risky = solve_risk_adjusted(&stats.covariance, &excess, tau, n)?;
    if (optimal_risky.sum() - 1.0).abs() > 1e-6 {
        return Err("Optimal risky weights do not sum to 1.".into());
    }
    Ok(optimal_risky)
}

/// Closed form through the inverse covariance
#[cfg(feature = "openblas")]
fn solve_risk_adjusted(
    cov: &Array2<f64>,
    excess: &Array1<f64>,
    tau: f64,
    n: usize,
) -> Result<Array1<f64>, Error> {
    let cov_inv: Array2<f64> = cov.clone().inv_into()?;
    let ones = Array1::<f64>::ones(n);
    let a = ones.dot(&cov_inv.dot(&ones));
    let b = ones.dot(&cov_inv.dot(excess));
    let lambda_multiplier = (b - 2.0 * tau) / a;
    Ok(cov_inv.dot(&(excess - lambda_multiplier)) / (2.0 * tau))
}

#[cfg(not(feature = "openblas"))]
fn solve_risk_adjusted(
    cov: &Array2<f64>,
    excess: &Array1<f64>,
    tau: f64,
    n: usize,
) -> Result<Array1<f64>, Error> {
    let x_equal = Array1::from_elem(n, 1.0 / n as f64);
    Ok(minimize_quadratic(
        &(cov * (2.0 * tau)),
        excess,
        x_equal,
        true,
    ))
}

/// Near-optimality method (Lolic, 2024).
/// Step 1: maximize utility μᵀx − ½τxᵀΣx, giving ε.
/// Step 2: minimize concentration xᵀx subject to utility ≥ θε.
/// Both subject to 1ᵀx = 1, and x ≥ 0 unless `short`.
pub fn near_optimal_weights(
    mean: &Array1<f64>,
    cov: &Array2<f64>,
    tau: f64,
    theta: f64,
    short: bool,
) -> Result<Array1<f64>, Error> {
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

/// Asset resampling (Lolic, 2024, Method Two): long-only MVO on `iterations` random
/// subsets of `m` assets, weights averaged. The draws are seeded, so results repeat.
pub fn resampled_weights(
    mean: &Array1<f64>,
    cov: &Array2<f64>,
    tau: f64,
    m: usize,
    iterations: usize,
) -> Result<Array1<f64>, Error> {
    let n = mean.len();
    if m == 0 || m > n || iterations == 0 {
        return Err("Need 1 <= m <= number of assets and at least one iteration.".into());
    }
    let mut state = 1u64;
    let mut order: Vec<usize> = (0..n).collect();
    let mut weights = Array1::<f64>::zeros(n);
    for _ in 0..iterations {
        // Partial Fisher-Yates shuffle with a linear congruential generator
        for i in 0..m {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            order.swap(i, i + (state >> 33) as usize % (n - i));
        }
        let subset = &order[..m];
        let sub_mean = Array1::from_iter(subset.iter().map(|&a| mean[a]));
        let sub_cov = Array2::from_shape_fn((m, m), |(i, j)| cov[[subset[i], subset[j]]]);
        let x_equal = Array1::from_elem(m, 1.0 / m as f64);
        let x = minimize_quadratic(&(sub_cov * tau), &sub_mean, x_equal, false);
        for (&asset, w) in subset.iter().zip(&x) {
            weights[asset] += w / iterations as f64;
        }
    }
    Ok(weights)
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
    fn risk_adjusted_two_assets() {
        let stats = PortfolioStats {
            assets: vec!["A".into(), "B".into()],
            mean_returns: Array1::from_vec(vec![0.1, 0.05]),
            covariance: Array2::from_diag(&Array1::from_vec(vec![0.04, 0.01])),
            returns_matrix: Array2::zeros((2, 0)),
        };
        let weights = risk_adjusted_weights(&stats, 0.0, 2.0).unwrap().to_vec();
        for (w, e) in weights.iter().zip([0.45, 0.55]) {
            assert!((w - e).abs() < 1e-9, "{weights:?}");
        }
    }

    // Expected: all 252 five-asset subsets averaged. 1000 draws scatter around it,
    // as the paper's Table 4 does.
    #[test]
    fn resampled_matches_enumeration() {
        let mean = Array1::from_vec(MEAN.to_vec());
        let cov = Array2::from_shape_fn((10, 10), |(i, j)| STD[i] * STD[j] * CORR[i][j]);
        let weights = resampled_weights(&mean, &cov, 3.0, 5, 1000).unwrap();
        let expected = [
            0.0321, 0.0454, 0.2496, 0.0709, 0.0082, 0.1833, 0.0537, 0.0855, 0.2702, 0.0011,
        ];
        for (w, e) in weights.iter().zip(expected) {
            assert!((w - e).abs() < 0.03, "{weights:?}");
        }
        assert!(resampled_weights(&mean, &cov, 3.0, 11, 10).is_err());
    }

    #[test]
    fn low_floor_gives_equal_weight() {
        assert_close(&weights(0.5, false), &[0.1; 10]);
    }
}
