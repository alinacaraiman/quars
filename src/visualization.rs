use plotters::prelude::*;
use std::error::Error;

use crate::optimization::{annual_to_daily_rate, OptimizationResults};


pub fn plot_efficient_frontier(results: &OptimizationResults, risk_free_rate: f64) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("efficient_frontier.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Identify bounding box for chart
    // x-axis: standard deviation (0..some max)
    // y-axis: return (some range)
    let max_std = results.frontier
        .iter()
        .map(|pt| pt.portfolio_std)
        .fold(0.0/0.0, f64::max);
    let max_ret = results.frontier
        .iter()
        .map(|pt| pt.expected_return)
        .fold(f64::MIN, f64::max);
    // Padding
    let x_max = max_std * 1.1;
    let y_max = max_ret * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Efficient Frontier", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..x_max, 0f64..y_max)?;

    chart.configure_mesh()
        .x_desc("Standard Deviation (Risk)")
        .y_desc("Expected Return")
        .draw()?;

    chart.draw_series(
        results.frontier.iter().map(|pt| {
            Circle::new((pt.portfolio_std, pt.expected_return), 3, &BLUE)
        })
    )?;

    let tang_x = results.optimal_risky_std;
    let tang_y = results.optimal_risky_return;
    chart.draw_series(std::iter::once(Circle::new((tang_x, tang_y), 5, RED)))?
        .label("Tangency Portfolio")
        .legend(|(x, y)| Circle::new((x, y), 5, RED));

    // Plot capital allocation line from risk-free (0, r_f) to tangency
    let daily_risk_free = annual_to_daily_rate(risk_free_rate);
    let cal_points = vec![
        (0.0, daily_risk_free),
        (tang_x, tang_y)
    ];
    chart.draw_series(LineSeries::new(cal_points, BLACK))?
        .label("Capital Allocation Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x+10, y)], BLACK));

    chart.configure_series_labels().draw()?;

    root.present()?;
    println!("Efficient frontier saved to efficient_frontier.png");
    Ok(())
}


pub fn plot_portfolio(
    asset_labels: &[String],
    weights: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    let root = BitMapBackend::new("portfolio.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    // Determine max weight for the y-axis
    let max_weight = weights.iter().cloned().fold(f64::NAN, f64::max).max(1.0); // ensure minimum range

    let mut chart = ChartBuilder::on(&root)
        .caption("Portfolio Weights", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..weights.len(), -max_weight * 1.2..max_weight * 1.2)?;

    chart
        .configure_mesh()
        .disable_mesh() // optional, for aesthetics
        .x_labels(weights.len())
        .x_label_formatter(&|x| {
            let idx = *x;
            if idx < asset_labels.len() {
                asset_labels[idx].clone()
            } else {
                "".to_string() // Out-of-range check
            }
        })
        .x_desc("Assets")
        .y_desc("Weight")
        .draw()?;

    // Draw a bar from (i, 0.0) to (i+1, weight)
    chart.draw_series(
        weights
            .iter()
            .enumerate()
            .map(|(i, &w)| Rectangle::new([(i, 0.0), (i + 1, w)], BLUE.filled())),
    )?;

    root.present()?;
    println!("Portfolio chart saved to portfolio.png");
    Ok(())
}


pub fn plot_return_distribution(
    returns: &Vec<f64>,
    var: f64,
    cvar: f64,
) -> Result<(), Box<dyn Error>> {
    // Define output file and create drawing area.
    let output_path = "portfolio_distribution.png";
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate min and max returns for the x-axis
    let min_return = returns.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_return = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Set number of bins for hist.
    let num_bins = 50;
    let bin_width = (max_return - min_return) / num_bins as f64;

    let mut bins = vec![0; num_bins];
    for r in returns {
        let mut bin = ((*r - min_return) / bin_width) as usize;
        if bin >= num_bins {
            bin = num_bins - 1;
        }
        bins[bin] += 1;
    }
    let max_count = bins.iter().cloned().max().unwrap_or(1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Portfolio Returns Distribution", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_return..max_return, 0..max_count)?;

    chart
        .configure_mesh()
        .x_desc("Return")
        .y_desc("Frequency")
        .draw()?;

    for (i, count) in bins.iter().enumerate() {
        let x0 = min_return + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0), (x1, *count)],
            BLUE.filled(),
        )))?;
    }

    // VaR line
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(var, 0), (var, max_count)],
            RED,
        )))?
        .label(format!("VaR(95%): {:.2}%", var * 100.0))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // CVaR line
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(cvar, 0), (cvar, max_count)],
            BLACK,
        )))?
        .label(format!("CVaR(95%): {:.2}%", cvar * 100.0))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    chart.configure_series_labels().border_style(BLACK).draw()?;

    root.present()?;
    println!("Portfolio returns distribution saved to {}", output_path);
    Ok(())
}
