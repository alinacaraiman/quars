use core::error;

use chrono::Local;
use utils::write_to_csv;

mod config;
mod data;
mod optimization;
mod portfolio;
mod utils;
mod visualization;
mod math;

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

    // Compute statistics
    let portfolio_stats = portfolio::calculate_portfolio_stats(&historical_data)
        .expect("Error computing portfolio stats");

    //Run optimization
    let results =
        optimization::optimize_portfolio(&portfolio_stats, 50, &settings.portofolio_optimization)
            .expect("Error in Markowitz optimization");

    // Show tangency portfolio
    println!(
        "Tangency Portfolio Weights = {:?}",
        results.optimal_risky_portfolio
    );
    println!(
        "Tangency Expected Return = {:.4}",
        results.optimal_risky_return
    );
    println!("Tangency Std Dev = {:.4}", results.optimal_risky_std);
    println!("Max Sharpe = {:.4}", results.max_sharpe);

    // Plot frontier
    visualization::plot_efficient_frontier(
        &results,
        settings.portofolio_optimization.risk_free_rate
    )?;
    // Plot portofolio weights
    visualization::plot_portfolio(&settings.data_api.tickers, &results.optimal_risky_portfolio)?;

    // Compute VaR & CVaR for tangency portfolio
    let tang_returns = portfolio::compute_portfolio_returns(
        &portfolio_stats.returns_matrix,
        &results.optimal_risky_portfolio,
    );
    let var_95 = portfolio::portfolio_var(&tang_returns, 0.95);
    let cvar_95 = portfolio::portfolio_cvar(&tang_returns, 0.95);

    println!("VaR(95%) = {:.2}%", var_95 * 100.0);
    println!("CVaR(95%) = {:.2}%", cvar_95 * 100.0);

    // Plot portfolio distribution and computed VaR and CVaR
    visualization::plot_return_distribution(&tang_returns, var_95, cvar_95)
        .expect("Failed to plot return distribution");
    Ok(())
}
