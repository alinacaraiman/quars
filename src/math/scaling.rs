use std::error::Error;

/// Hurst exponent by rescaled range analysis (Hurst, 1951): the slope of log R/S
/// against log window size over `scales`.
/// 0.5 for uncorrelated increments, biased upwards in short series.
pub fn hurst_rs(series: &[f64], scales: &[usize]) -> Result<f64, Box<dyn Error>> {
    scaling_exponent(series, scales, |window| {
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let (mut z, mut min, mut max, mut ss) = (0.0, f64::MAX, f64::MIN, 0.0);
        for x in window {
            z += x - mean;
            min = min.min(z);
            max = max.max(z);
            ss += (x - mean).powi(2);
        }
        (max - min) / (ss / window.len() as f64).sqrt()
    })
}

/// Detrended fluctuation analysis (Peng et al., 1994) with linear detrending: the slope
/// of the log fluctuation against log window size over `scales`.
/// 0.5 for uncorrelated noise, 1.5 for a random walk.
pub fn dfa(series: &[f64], scales: &[usize]) -> Result<f64, Box<dyn Error>> {
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let mut sum = 0.0;
    let profile: Vec<f64> = series
        .iter()
        .map(|x| {
            sum += x - mean;
            sum
        })
        .collect();
    // Mean squared residual of a least-squares line through the window
    let exponent = scaling_exponent(&profile, scales, |window| {
        let n = window.len() as f64;
        let (t_mean, y_mean) = ((n - 1.0) / 2.0, window.iter().sum::<f64>() / n);
        let (mut sty, mut stt, mut syy) = (0.0, 0.0, 0.0);
        for (t, y) in window.iter().enumerate() {
            let (dt, dy) = (t as f64 - t_mean, y - y_mean);
            sty += dt * dy;
            stt += dt * dt;
            syy += dy * dy;
        }
        (syy - sty * sty / stt) / n
    })?;
    Ok(0.5 * exponent)
}

/// Slope of log mean `stat` over non-overlapping windows against log window size
fn scaling_exponent(
    series: &[f64],
    scales: &[usize],
    stat: impl Fn(&[f64]) -> f64,
) -> Result<f64, Box<dyn Error>> {
    if scales.len() < 2 || scales.iter().any(|&s| s < 4 || s > series.len()) {
        return Err("Need at least two scales, each between 4 and the series length.".into());
    }
    let points: Vec<(f64, f64)> = scales
        .iter()
        .map(|&s| {
            let values: Vec<f64> = series
                .chunks_exact(s)
                .map(&stat)
                .filter(|v| v.is_finite())
                .collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            ((s as f64).ln(), mean.ln())
        })
        .collect();
    let n = points.len() as f64;
    let x_mean = points.iter().map(|p| p.0).sum::<f64>() / n;
    let y_mean = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxy: f64 = points.iter().map(|p| (p.0 - x_mean) * (p.1 - y_mean)).sum();
    let sxx: f64 = points.iter().map(|p| (p.0 - x_mean).powi(2)).sum();
    let slope = sxy / sxx;
    if !slope.is_finite() {
        return Err("Series must be finite and not constant, scales distinct.".into());
    }
    Ok(slope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::test_noise;

    const SCALES: [usize; 4] = [16, 32, 64, 128];

    fn walk() -> Vec<f64> {
        let mut sum = 0.0;
        test_noise(1024)
            .iter()
            .map(|x| {
                sum += x;
                sum
            })
            .collect()
    }

    // Reference values from Python `nolds` (dfa, hurst_rs without corrections)
    #[test]
    fn noise() {
        let noise = test_noise(1024);
        assert!((dfa(&noise, &SCALES).unwrap() - 0.492316714).abs() < 1e-6);
        assert!((hurst_rs(&noise, &SCALES).unwrap() - 0.563661905).abs() < 1e-6);
    }

    #[test]
    fn random_walk() {
        assert!((dfa(&walk(), &SCALES).unwrap() - 1.424350574).abs() < 1e-6);
        assert!((hurst_rs(&walk(), &SCALES).unwrap() - 1.024503729).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(dfa(&test_noise(64), &[16]).is_err());
        assert!(dfa(&test_noise(64), &[16, 128]).is_err());
        assert!(hurst_rs(&[1.0; 64], &[8, 16]).is_err());
    }
}
