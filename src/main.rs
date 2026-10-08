use core::error;

use chrono::Local;
use quars::utils::write_to_csv;
use quars::{config, data, optimization, portfolio, visualization};

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    let settings = config::Settings::new().expect("Failed to load configuration");
    let historical_data = data::fetch_data(&settings).await.expect("Data fetch error");
    let today = Local::now().format("%Y-%m-%d").to_string();
    let output_path = format!(
        "data/raw/{}/hist_data_{}.csv",
        today, settings.data_api.source
    );
    write_to_csv(&historical_data, &output_path).expect("Failed to write CSV");

    let stats = portfolio::calculate_portfolio_stats(&historical_data)
        .expect("Error computing portfolio stats");
    let po = &settings.portofolio_optimization;
    // params: tau, theta (default: the paper's 0.95), resampling subset size (default: half the assets)
    let tau = *po.params.first().ok_or("params must start with tau")?;
    let theta = po.params.get(1).copied().unwrap_or(0.95);
    let m = po
        .params
        .get(2)
        .map_or(stats.assets.len().div_ceil(2), |&m| m as usize);
    let portfolios = optimization::optimize_portfolios(&stats, po.risk_free_rate, tau, theta, m)
        .expect("Error in portfolio optimization");

    let risk_free = optimization::annual_to_daily_rate(po.risk_free_rate);
    for p in &portfolios {
        println!(
            "{:<20} return {:.4}  std {:.4}  sharpe {:.4}",
            p.name,
            p.expected_return,
            p.std,
            (p.expected_return - risk_free) / p.std
        );
    }
    let selected = portfolios
        .iter()
        .find(|p| p.name == po.sub_method)
        .ok_or("Unknown sub_method")?;
    println!("{} weights = {:?}", selected.name, selected.weights);

    let frontier = optimization::efficient_frontier(&stats, 50);
    visualization::plot_efficient_frontier(&stats, &frontier, &portfolios, po.risk_free_rate)?;
    visualization::plot_portfolio(&stats.assets, &portfolios)?;
    if let Ok(corr) = quars::math::correlation::correlation(&stats.covariance) {
        visualization::plot_correlation_tree(&stats.assets, &corr)?;
    }

    let returns = portfolio::compute_portfolio_returns(&stats.returns_matrix, &selected.weights);
    let var_95 = portfolio::portfolio_var(&returns, 0.95).ok_or("No portfolio returns")?;
    let cvar_95 = portfolio::portfolio_cvar(&returns, 0.95).ok_or("No portfolio returns")?;
    println!("VaR(95%) = {:.2}%", var_95 * 100.0);
    println!("CVaR(95%) = {:.2}%", cvar_95 * 100.0);
    visualization::plot_return_distribution(selected.name, &returns, var_95, cvar_95)?;
    Ok(())
}
