use plotters::coord::Shift;
use plotters::prelude::*;
use std::error::Error;

use crate::optimization::{annual_to_daily_rate, Portfolio};
use crate::portfolio::PortfolioStats;

const SURFACE: RGBColor = RGBColor(252, 252, 251);
const INK: RGBColor = RGBColor(82, 81, 78);
const GRID: RGBColor = RGBColor(226, 225, 221);
// One per method, in `optimize_portfolios` order
const SERIES: [RGBColor; 5] = [
    RGBColor(42, 120, 214),
    RGBColor(237, 161, 0),
    RGBColor(27, 175, 122),
    RGBColor(74, 58, 167),
    RGBColor(227, 73, 72),
];
const VAR: RGBColor = RGBColor(208, 59, 59);

type Area<'a> = DrawingArea<BitMapBackend<'a>, Shift>;

fn drawing_area(path: &str, size: (u32, u32)) -> Result<Area<'_>, Box<dyn Error>> {
    let root = BitMapBackend::new(path, size).into_drawing_area();
    root.fill(&SURFACE)?;
    Ok(root)
}

// Shape varies with the method too, so colour is not the only cue
fn marker<'a, C: Clone + 'static>(i: usize, at: C, r: i32) -> DynElement<'a, BitMapBackend<'a>, C> {
    let color = SERIES[i % SERIES.len()];
    match i % 3 {
        0 => Circle::new(at, r, color.filled()).into_dyn(),
        1 => TriangleMarker::new(at, r + 1, color.filled()).into_dyn(),
        _ => Cross::new(at, r, color.stroke_width(3)).into_dyn(),
    }
}

/// Long-only frontier with the assets, the capital allocation line and every method
pub fn plot_efficient_frontier(
    stats: &PortfolioStats,
    frontier: &[(f64, f64)],
    portfolios: &[Portfolio],
    risk_free_rate: f64,
) -> Result<(), Box<dyn Error>> {
    let root = drawing_area("efficient_frontier.png", (900, 600))?;
    let pct = |(std, ret): (f64, f64)| (std * 100.0, ret * 100.0);
    let frontier: Vec<_> = frontier.iter().copied().map(pct).collect();
    let assets: Vec<_> = (0..stats.assets.len())
        .map(|i| pct((stats.covariance[[i, i]].sqrt(), stats.mean_returns[i])))
        .collect();
    let methods: Vec<_> = portfolios
        .iter()
        .map(|p| pct((p.std, p.expected_return)))
        .collect();
    let risk_free = annual_to_daily_rate(risk_free_rate) * 100.0;
    let max_sharpe = frontier
        .iter()
        .filter(|p| p.0 > 0.0)
        .map(|p| (p.1 - risk_free) / p.0)
        .fold(f64::MIN, f64::max);

    let points = || frontier.iter().chain(&assets).chain(&methods);
    let x_max = points().map(|p| p.0).fold(0.0, f64::max) * 1.1;
    let y_max = points().map(|p| p.1).fold(risk_free, f64::max) * 1.1;
    let y_min = points().map(|p| p.1).fold(0.0, f64::min) * 1.1;

    let cal_end = x_max.min((y_max - risk_free) / max_sharpe);

    let mut chart = ChartBuilder::on(&root)
        .caption("Efficient Frontier", ("sans-serif", 26))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..x_max, y_min..y_max)?;
    chart
        .configure_mesh()
        .light_line_style(TRANSPARENT)
        .bold_line_style(GRID)
        .axis_style(INK)
        .x_desc("Standard Deviation (%)")
        .y_desc("Expected Return (%)")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            [
                (0.0, risk_free),
                (cal_end, risk_free + max_sharpe * cal_end),
            ],
            INK.mix(0.5),
        ))?
        .label("Capital allocation line")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], INK.mix(0.5)));
    chart
        .draw_series(LineSeries::new(frontier, INK.stroke_width(2)))?
        .label("Efficient frontier (long-only)")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], INK.stroke_width(2)));
    chart.draw_series(assets.iter().zip(&stats.assets).map(|(&at, name)| {
        EmptyElement::at(at)
            + Circle::new((0, 0), 4, INK.stroke_width(1))
            + Text::new(
                name.clone(),
                (7, -14),
                ("sans-serif", 13).into_font().color(&INK),
            )
    }))?;
    for (i, (p, &at)) in portfolios.iter().zip(&methods).enumerate() {
        chart
            .draw_series(std::iter::once(marker(i, at, 7)))?
            .label(p.name)
            .legend(move |(x, y)| marker(i, (x + 10, y), 5));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(SURFACE.mix(0.9))
        .border_style(GRID)
        .draw()?;
    root.present()?;
    println!("Efficient frontier saved to efficient_frontier.png");
    Ok(())
}

/// Weights of every method, grouped by asset
pub fn plot_portfolio(assets: &[String], portfolios: &[Portfolio]) -> Result<(), Box<dyn Error>> {
    let root = drawing_area("portfolio.png", (900, 500))?;
    let weights = || {
        portfolios
            .iter()
            .flat_map(|p| &p.weights)
            .map(|w| w * 100.0)
    };
    let y_max = weights().fold(0.0, f64::max) * 1.1;
    let y_min = weights().fold(0.0, f64::min) * 1.1;
    let n = assets.len() as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption("Portfolio Weights", ("sans-serif", 26))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(-0.5..n - 0.5, y_min..y_max)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .light_line_style(TRANSPARENT)
        .bold_line_style(GRID)
        .axis_style(INK)
        .x_labels(assets.len())
        .x_label_formatter(&|x| match assets.get(x.round() as usize) {
            Some(asset) if (x - x.round()).abs() < 1e-9 => asset.clone(),
            _ => String::new(),
        })
        .x_desc("Assets")
        .y_desc("Weight (%)")
        .draw()?;

    let width = 0.8 / portfolios.len() as f64;
    for (i, p) in portfolios.iter().enumerate() {
        let color = SERIES[i % SERIES.len()];
        chart
            .draw_series(p.weights.iter().enumerate().map(|(asset, w)| {
                let x = asset as f64 - 0.4 + i as f64 * width;
                let mut bar = Rectangle::new([(x, 0.0), (x + width, w * 100.0)], color.filled());
                bar.set_margin(0, 0, 1, 1);
                bar
            }))?
            .label(p.name)
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 12, y + 5)], color.filled()));
    }
    chart.draw_series(LineSeries::new([(-0.5, 0.0), (n - 0.5, 0.0)], INK))?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(SURFACE.mix(0.9))
        .border_style(GRID)
        .draw()?;
    root.present()?;
    println!("Portfolio chart saved to portfolio.png");
    Ok(())
}

pub fn plot_return_distribution(
    name: &str,
    returns: &[f64],
    var: f64,
    cvar: f64,
) -> Result<(), Box<dyn Error>> {
    // Define output file and create drawing area.
    let output_path = "portfolio_distribution.png";
    let root = drawing_area(output_path, (800, 600))?;

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
        .caption(
            format!("Portfolio Returns Distribution ({})", name),
            ("sans-serif", 26),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_return..max_return, 0..max_count)?;

    chart
        .configure_mesh()
        .light_line_style(TRANSPARENT)
        .bold_line_style(GRID)
        .axis_style(INK)
        .x_desc("Return")
        .y_desc("Frequency")
        .draw()?;

    for (i, count) in bins.iter().enumerate() {
        let x0 = min_return + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0), (x1, *count)],
            SERIES[0].filled(),
        )))?;
    }

    // VaR line
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(var, 0), (var, max_count)],
            VAR.stroke_width(2),
        )))?
        .label(format!("VaR(95%): {:.2}%", var * 100.0))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], VAR.stroke_width(2)));

    // CVaR line
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(cvar, 0), (cvar, max_count)],
            BLACK.stroke_width(2),
        )))?
        .label(format!("CVaR(95%): {:.2}%", cvar * 100.0))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(2)));

    chart
        .configure_series_labels()
        .background_style(SURFACE.mix(0.9))
        .border_style(GRID)
        .draw()?;

    root.present()?;
    println!("Portfolio returns distribution saved to {}", output_path);
    Ok(())
}
