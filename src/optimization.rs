use crate::{config::PortofolioOptimization, portfolio::PortfolioStats};
use ndarray::{Array1, Array2};
use ndarray_linalg::InverseInto;
use std::error::Error;

// Optim. method Enum for Mean Variance Optimization
pub enum MvoOptMethod {
    // Maximize risk-adjusted return
    RiskAdjusted { tau: f64 },
    // Near-optimality method, minimize concentration of weights after computing standart MVO
    NearOptimal { tau: f64, theta: f64 },
}

impl MvoOptMethod {
    pub fn from_config(portofolio_optimization_config: &PortofolioOptimization) -> Self {
        match portofolio_optimization_config.sub_method.as_str() {
            "risk-adjusted" => Self::RiskAdjusted {
                tau: portofolio_optimization_config.params[0],
            },
            "near-optimal" => Self::NearOptimal {
                tau: portofolio_optimization_config.params[0],
                theta: portofolio_optimization_config.params[1],
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
        MvoOptMethod::NearOptimal { theta, tau } => {
            optimize_near_optimal(stats, po.risk_free_rate, tau, theta, n_points)
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
    let optimal_risky_return = mean.dot(&optimal_risky);
    let variance_risky = optimal_risky.dot(&cov.dot(&optimal_risky));
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
    Ok(OptimizationResults {
        frontier,
        optimal_risky_portfolio: optimal_risky.to_vec(),
        optimal_risky_return,
        optimal_risky_std,
        max_sharpe,
    })
}

/// Near-optimality method implementation.
/// Step 1: Compute classic MVO
/// Use a simple closed-form approximation assuming an unconstrained problem:
/// x_mvo ∝ Σ⁻¹ * μ, then normalize so that 1ᵀx = 1.
fn optimize_near_optimal(
    stats: &PortfolioStats,
    risk_free_rate: f64,
    tau: f64,
    theta: f64,
    n_points: usize,
) -> Result<OptimizationResults, Box<dyn Error>> {
    let n = stats.assets.len();
    let mean = stats.mean_returns.clone();
    let cov = stats.covariance.clone();
    let cov_inv: Array2<f64> = cov.clone().inv_into()?;
    let daily_risk_free = annual_to_daily_rate(risk_free_rate);
    let x_mvo_unnorm = cov_inv.dot(&mean);
    let sum_x = x_mvo_unnorm.sum();
    let x_mvo = x_mvo_unnorm.mapv(|val| val / sum_x);
    // Compute ex ante utility based on standart MVO: ε = μᵀx_mvo - ½γ x_mvoᵀΣx_mvo
    let epsilon = mean.dot(&x_mvo) - 0.5 * tau * x_mvo.dot(&cov.dot(&x_mvo));
    dbg!(&mean);
    dbg!(&cov);
    // Minimize concentration (xᵀx)
    // As noted in the paper, near-optimality approach is not strictly linear and a solution for the
    // minimization of the objective function, such that the constraining portofolio utility is greater
    // or equal to a preset percentace (Theta) of the standard optimized portfolio.
    // A computationally cheap and easy initial solution to that is to initialize an equal-weight portofolio and blend with the
    // classic MVO one, until the utility is at least θε
    // Cons: depending on the picked assets, most of the cases will lead to an equal-weighted portofolio, since the variance in
    // weights due to the the brute-forced alpha is very low.
    // Convex quadratic solver could lead to optimal weights.
    let x_equal = Array1::from_elem(n, 1.0 / n as f64);
    let mut best_blend = x_mvo.clone();

    // Initialize concentration with the classic mvo weights
    let mut weights_concentration = x_mvo.t().dot(&x_mvo);

    // Set an arbitrary parameter alpha_0 equal to 1, minimizing at each iteration
    // TODO, implement convex quadratic optimization
    for i in 0..=100 {
        let alpha = i as f64 / 100.0;
        let x_blend = &x_equal * (1.0 - alpha) + &x_mvo * alpha;
        let utility = mean.dot(&x_blend) - 0.5 * tau * x_blend.dot(&cov.dot(&x_blend));
        // (x_blend.sum() - 1.0).abs() < 1e-6: for floating point tolerance
        if utility >= theta * epsilon
            && (x_blend.sum() - 1.0).abs() < 1e-6
        {
            // xᵀx
            let concentration = x_blend.t().dot(&x_blend);
            if concentration < weights_concentration {
                print!("Concentraton: {:?}, best x blend {:?}, utility {:?}", concentration, x_blend, utility);
                weights_concentration = concentration;
                best_blend = x_blend;
            }
        }
    }

    // repeat for inverse case
    for i in 0..=100 {
        let alpha = i as f64 / 100.0;
        let x_blend = &x_equal * alpha + &x_mvo * (1.0 - alpha);
        let utility = mean.dot(&x_blend) - 0.5 * tau * x_blend.dot(&cov.dot(&x_blend));
        // (x_blend.sum() - 1.0).abs() < 1e-6: for floating point tolerance
        if utility >= theta * epsilon
            && (x_blend.sum() - 1.0).abs() < 1e-6
        {
            // xᵀx
            let concentration = x_blend.t().dot(&x_blend);
            if concentration < weights_concentration {
                print!("Concentraton: {:?}, best x blend {:?}, utility {:?}", concentration, x_blend, utility);
                weights_concentration = concentration;
                best_blend = x_blend;
            }
        }
    }
    let optimal_risky = best_blend;
    let optimal_risky_return = mean.dot(&optimal_risky);
    let variance_risky = optimal_risky.dot(&cov.dot(&optimal_risky));
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

    Ok(OptimizationResults {
        frontier,
        optimal_risky_portfolio: optimal_risky.to_vec(),
        optimal_risky_return,
        optimal_risky_std,
        max_sharpe,
    })
}

pub fn annual_to_daily_rate(r_annual: f64) -> f64 {
    (1.0 + r_annual).powf(1.0 / 252.0) - 1.0
}