use crate::{
    config::PortofolioOptimization, math::optimization::minimize_quadratic,
    portfolio::PortfolioStats,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::InverseInto;
use std::error::Error;

// Optim. method Enum for Mean Variance Optimization
pub enum MvoOptMethod {
    // Maximize risk-adjusted return
    RiskAdjusted { tau: f64 },
    // Near-optimality method, minimize concentration of weights after computing standart MVO
    NearOptimal { tau: f64, theta: f64, short: bool },
}

impl MvoOptMethod {
    pub fn from_config(portofolio_optimization_config: &PortofolioOptimization) -> Self {
        match portofolio_optimization_config.sub_method.as_str() {
            "risk-adjusted" => Self::RiskAdjusted {
                tau: portofolio_optimization_config.params[0],
            },
            sub_method @ ("near-optimal" | "near-optimal-short") => Self::NearOptimal {
                tau: portofolio_optimization_config.params[0],
                theta: portofolio_optimization_config.params[1],
                short: sub_method == "near-optimal-short",
            },
            _ => Self::RiskAdjusted { tau: 0.3 },
        }
    }
}

#[derive(Clone, Debug)]
pub struct FrontierPoint {
    risk_free_weight: f64,
    risky_weights: Vec<f64>,
    pub expected_return: f64,
    pub portfolio_std: f64,
    sharpe_ratio: f64,
}

#[derive(Debug)]
pub struct OptimizationResults {
    pub frontier: Vec<FrontierPoint>,
    // The optimal risky asset weights
    pub optimal_risky_portfolio: Vec<f64>,
    // Expected return of the tangency portfolio
    pub optimal_risky_return: f64,
    pub optimal_risky_std: f64,
    pub max_sharpe: f64,
}

pub fn optimize_portfolio(
    stats: &PortfolioStats,
    n_points: usize,
    po: &PortofolioOptimization,
) -> Result<OptimizationResults, Box<dyn Error>> {
    let opt_method = MvoOptMethod::from_config(po);
    match opt_method {
        MvoOptMethod::RiskAdjusted { tau } => {
            optimize_risk_adjusted(stats, po.risk_free_rate, tau, n_points)
        }
        MvoOptMethod::NearOptimal { theta, tau, short } => {
            let weights =
                near_optimal_weights(&stats.mean_returns, &stats.covariance, tau, theta, short)?;
            Ok(build_results(stats, po.risk_free_rate, weights, n_points))
        }
    }
}

fn optimize_risk_adjusted(
    stats: &PortfolioStats,
    risk_free_rate: f64,
    tau: f64,
    n_points: usize,
) -> Result<OptimizationResults, Box<dyn Error>> {
    let n = stats.assets.len();
    let mean = stats.mean_returns.clone();
    let cov = stats.covariance.clone();
    let daily_risk_free = annual_to_daily_rate(risk_free_rate);
    let cov_inv: Array2<f64> = cov.clone().inv_into()?;
    let ones = Array1::<f64>::ones(n);
    let excess = &mean - ones.mapv(|_| daily_risk_free);
    let A = ones.dot(&cov_inv.dot(&ones));
    let B = ones.dot(&cov_inv.dot(&excess));
    let lambda_multiplier = (B - 2.0 * tau) / A;
    let factor = 1.0 / (2.0 * tau);
    let optimal_risky = cov_inv.dot(&(&excess - ones.mapv(|_| lambda_multiplier))) * factor;
    let sum_weights = optimal_risky.sum();
    if (sum_weights - 1.0).abs() > 1e-6 {
        return Err("Optimal risky weights do not sum to 1.".into());
    }
    Ok(build_results(stats, risk_free_rate, optimal_risky, n_points))
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

fn build_results(
    stats: &PortfolioStats,
    risk_free_rate: f64,
    optimal_risky: Array1<f64>,
    n_points: usize,
) -> OptimizationResults {
    let daily_risk_free = annual_to_daily_rate(risk_free_rate);
    let optimal_risky_return = stats.mean_returns.dot(&optimal_risky);
    let variance_risky = optimal_risky.dot(&stats.covariance.dot(&optimal_risky));
    let optimal_risky_std = variance_risky.sqrt();
    let max_sharpe = (optimal_risky_return - daily_risk_free) / optimal_risky_std;
    let max_leverage = 2.0;
    let lambda_step = max_leverage / (n_points as f64 - 1.0);
    let mut frontier = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let leverage = i as f64 * lambda_step;
        let risk_free_weight = 1.0 - leverage;
        let scaled_risky: Vec<f64> = optimal_risky.mapv(|w| leverage * w).to_vec();
        let portfolio_return = daily_risk_free + leverage * (optimal_risky_return - daily_risk_free);
        let portfolio_std = leverage * optimal_risky_std;
        let sharpe_ratio = if leverage > 0.0 {
            (portfolio_return - daily_risk_free) / portfolio_std
        } else {
            0.0
        };
        frontier.push(FrontierPoint {
            risk_free_weight,
            risky_weights: scaled_risky,
            expected_return: portfolio_return,
            portfolio_std,
            sharpe_ratio,
        });
    }
    OptimizationResults {
        frontier,
        optimal_risky_portfolio: optimal_risky.to_vec(),
        optimal_risky_return,
        optimal_risky_std,
        max_sharpe,
    }
}

pub fn annual_to_daily_rate(r_annual: f64) -> f64 {
    (1.0 + r_annual).powf(1.0 / 252.0) - 1.0
}
#[cfg(test)]
mod tests {
    use super::*;

    // Capital markets assumptions of Lolic (2024), Appendix A
    const MEAN: [f64; 10] = [0.085, 0.09, 0.1, 0.095, 0.04, 0.055, 0.05, 0.07, 0.075, 0.075];
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
        let expected = [0.0669, 0.0739, 0.2055, 0.0986, 0.0166, 0.147, 0.0985, 0.1145, 0.1786, 0.0];
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
