use super::two_sided_p;
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bds {
    /// Standard normal under independence
    pub statistic: f64,
    /// Two-sided p-value
    pub p: f64,
}

/// BDS test (Brock, Dechert, Scheinkman & LeBaron, 1996) of the null that `series` is
/// i.i.d., against dependence of any kind, linear or nonlinear.
/// `m` is the embedding dimension and `epsilon_std` the distance threshold in standard
/// deviations of the series, commonly 0.5 to 2. Compares all pairs, so O(n²) time.
pub fn bds(series: &[f64], m: usize, epsilon_std: f64) -> Result<Bds, Box<dyn Error>> {
    let n = series.len();
    if m < 2 || n < m + 2 {
        return Err("Need m >= 2 and at least m + 2 observations.".into());
    }
    let nf = n as f64;
    let mean = series.iter().sum::<f64>() / nf;
    let std = (series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
    let epsilon = epsilon_std * std;
    if !(epsilon > 0.0 && epsilon.is_finite()) {
        return Err("Series must be finite and not constant, epsilon_std positive.".into());
    }
    let close = |i: usize, j: usize| (series[i] - series[j]).abs() < epsilon;
    // Share of pairs from `first` on whose `dim` consecutive values are all close
    let correlation_sum = |first: usize, dim: usize| {
        let end = n - dim + 1;
        let mut hits = 0u64;
        for i in first..end {
            for j in i + 1..end {
                hits += (0..dim).all(|k| close(i + k, j + k)) as u64;
            }
        }
        let points = (end - first) as f64;
        hits as f64 / (points * (points - 1.0) / 2.0)
    };

    let c = correlation_sum(0, 1);
    let neighbours: Vec<f64> = (0..n)
        .map(|i| (0..n).filter(|&j| close(i, j)).count() as f64)
        .collect();
    let k = (neighbours.iter().map(|v| v * v).sum::<f64>() - 3.0 * neighbours.iter().sum::<f64>()
        + 2.0 * nf)
        / (nf * (nf - 1.0) * (nf - 2.0));
    let mf = m as f64;
    let cross: f64 = (1..m)
        .map(|j| k.powi((m - j) as i32) * c.powi(2 * j as i32))
        .sum();
    let variance = 4.0
        * (k.powi(m as i32) + 2.0 * cross + (mf - 1.0).powi(2) * c.powi(2 * m as i32)
            - mf * mf * k * c.powi(2 * m as i32 - 2));

    let effect = correlation_sum(0, m) - correlation_sum(m - 1, 1).powi(m as i32);
    let statistic = (nf - mf + 1.0).sqrt() * effect / variance.sqrt();
    if !statistic.is_finite() {
        return Err("BDS statistic is undefined for this series and epsilon_std.".into());
    }
    Ok(Bds {
        statistic,
        p: two_sided_p(statistic),
    })
}

/// 0–1 test for chaos (Gottwald & Melbourne, 2009): near 0 for regular dynamics and
/// near 1 for chaotic dynamics of a deterministic system.
/// Random noise also gives a value near 1, so this does not tell chaos from noise.
pub fn zero_one_test(series: &[f64]) -> Result<f64, Box<dyn Error>> {
    let n = series.len();
    let n_cut = n / 10;
    if n_cut < 2 {
        return Err("Need at least 20 observations.".into());
    }
    let m = n - n_cut;
    let mean = series.iter().sum::<f64>() / n as f64;
    let ss: f64 = series.iter().map(|x| (x - mean).powi(2)).sum();
    if !(ss > 0.0 && ss.is_finite()) {
        return Err("Series must be finite and not constant.".into());
    }
    let mut ks: Vec<f64> = (0..100)
        .map(|i| {
            let c = PI / 5.0 + (i as f64 + 0.5) / 100.0 * 3.0 * PI / 5.0;
            // Translation variables driven by the series
            let (mut p, mut q) = (0.0, 0.0);
            let path: Vec<(f64, f64)> = series
                .iter()
                .enumerate()
                .map(|(j, x)| {
                    let angle = (j + 1) as f64 * c;
                    p += x * angle.cos();
                    q += x * angle.sin();
                    (p, q)
                })
                .collect();
            // Mean square displacement without its oscillating term, against the lag
            let points: Vec<(f64, f64)> = (1..=n_cut)
                .map(|lag| {
                    let displacement = (0..m)
                        .map(|j| {
                            (path[j + lag].0 - path[j].0).powi(2)
                                + (path[j + lag].1 - path[j].1).powi(2)
                        })
                        .sum::<f64>()
                        / m as f64;
                    let oscillation =
                        mean * mean * (1.0 - (lag as f64 * c).cos()) / (1.0 - c.cos());
                    (lag as f64, displacement - oscillation)
                })
                .collect();
            correlation(&points)
        })
        .collect();
    ks.sort_by(f64::total_cmp);
    Ok(0.5 * (ks[49] + ks[50]))
}

fn correlation(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let x_mean = points.iter().map(|p| p.0).sum::<f64>() / n;
    let y_mean = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in points {
        sxy += (x - x_mean) * (y - y_mean);
        sxx += (x - x_mean).powi(2);
        syy += (y - y_mean).powi(2);
    }
    sxy / (sxx * syy).sqrt()
}

/// Node degrees of the horizontal visibility graph (Luque et al., 2009): two observations
/// are linked when every observation between them is smaller than both.
pub fn visibility_degrees(series: &[f64]) -> Vec<usize> {
    let mut degrees = vec![0; series.len()];
    // Earlier observations still visible from the right, in decreasing order
    let mut visible: Vec<usize> = Vec::new();
    for (j, &x) in series.iter().enumerate() {
        while let Some(&i) = visible.last() {
            degrees[i] += 1;
            degrees[j] += 1;
            if series[i] > x {
                break;
            }
            visible.pop();
            if series[i] == x {
                break;
            }
        }
        visible.push(j);
    }
    degrees
}

/// Kullback-Leibler divergence of the visibility degree distribution from the exact one
/// of an i.i.d. series, P(k) = (1/3)(2/3)^(k − 2). Near 0 for uncorrelated noise.
/// A distance, not a p-value: degrees of nearby observations are dependent.
pub fn visibility_divergence(series: &[f64]) -> Result<f64, Box<dyn Error>> {
    let degrees: Vec<usize> = visibility_degrees(series)
        .into_iter()
        .filter(|&k| k >= 2)
        .collect();
    if degrees.is_empty() {
        return Err("Need at least 3 observations.".into());
    }
    let max = degrees.iter().copied().max().unwrap_or(2);
    Ok((2..=max)
        .map(|k| {
            let share = degrees.iter().filter(|&&d| d == k).count() as f64 / degrees.len() as f64;
            let iid = (2.0f64 / 3.0).powi(k as i32 - 2) / 3.0;
            if share > 0.0 {
                share * (share / iid).ln()
            } else {
                0.0
            }
        })
        .sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::test_noise;

    fn logistic(r: f64, n: usize) -> Vec<f64> {
        let mut x = 0.3;
        let map: Vec<f64> = (0..n + 100)
            .map(|_| {
                x = r * x * (1.0 - x);
                x
            })
            .collect();
        map[100..].to_vec()
    }

    // Reference values from Python `statsmodels.tsa.stattools.bds`
    #[test]
    fn bds_matches_reference() {
        let cases = [
            (test_noise(400), 2, -0.160980403, 0.872108835),
            (test_noise(400), 3, -0.309800821, 0.756712427),
            (logistic(3.97, 400), 2, 6.991095365, 0.0),
        ];
        for (series, m, statistic, p) in cases {
            let r = bds(&series, m, 1.5).unwrap();
            assert!(
                (r.statistic - statistic).abs() < 1e-6 && (r.p - p).abs() < 1e-6,
                "{r:?}"
            );
        }
    }

    #[test]
    fn bds_rejects_bad_input() {
        assert!(bds(&test_noise(64), 1, 1.5).is_err());
        assert!(bds(&[1.0; 64], 2, 1.5).is_err());
        assert!(bds(&test_noise(64), 2, 0.0).is_err());
    }

    #[test]
    fn zero_one_separates_regular_from_chaotic() {
        assert!((zero_one_test(&logistic(3.55, 1000)).unwrap() + 0.005684420).abs() < 1e-6);
        assert!((zero_one_test(&logistic(3.97, 1000)).unwrap() - 0.998316278).abs() < 1e-6);
        assert!(zero_one_test(&[1.0; 100]).is_err());
    }

    #[test]
    fn visibility_degrees_match_brute_force() {
        let mut series = test_noise(200);
        series.extend([1.0, 0.0, 1.0, 1.0, 2.0, 0.5, 2.0]);
        let n = series.len();
        let mut expected = vec![0; n];
        for i in 0..n {
            for j in i + 1..n {
                if (i + 1..j).all(|k| series[k] < series[i].min(series[j])) {
                    expected[i] += 1;
                    expected[j] += 1;
                }
            }
        }
        assert_eq!(visibility_degrees(&series), expected);
    }

    #[test]
    fn visibility_divergence_is_small_for_noise() {
        let noise = visibility_divergence(&test_noise(4096)).unwrap();
        let chaotic = visibility_divergence(&logistic(3.97, 4096)).unwrap();
        assert!(noise < 0.01 && chaotic > 10.0 * noise);
    }
}
