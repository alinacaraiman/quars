use std::error::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceRatio {
    pub vr: f64,
    /// Test statistic under homoskedasticity
    pub z: f64,
    /// Heteroskedasticity-robust test statistic
    pub z_robust: f64,
}

/// Lo-MacKinlay (1988) variance ratio of `q`-period to one-period returns,
/// with overlapping observations and bias correction.
/// Under a random walk `vr` is 1 and both statistics are standard normal.
pub fn variance_ratio(returns: &[f64], q: usize) -> Result<VarianceRatio, Box<dyn Error>> {
    let n = returns.len();
    if q < 2 || n <= q {
        return Err("Need q >= 2 and more than q returns.".into());
    }
    let (nf, qf) = (n as f64, q as f64);
    let mu = returns.iter().sum::<f64>() / nf;
    let sq: Vec<f64> = returns.iter().map(|r| (r - mu).powi(2)).collect();
    let ss: f64 = sq.iter().sum();
    if !(ss > 0.0 && ss.is_finite()) {
        return Err("Returns must be finite with non-zero variance.".into());
    }

    let var_1 = ss / (nf - 1.0);
    let m = qf * (nf - qf + 1.0) * (1.0 - qf / nf);
    let var_q = returns
        .windows(q)
        .map(|w| (w.iter().sum::<f64>() - qf * mu).powi(2))
        .sum::<f64>()
        / m;
    let vr = var_q / var_1;

    let phi = 2.0 * (2.0 * qf - 1.0) * (qf - 1.0) / (3.0 * qf * nf);
    let theta: f64 = (1..q)
        .map(|j| {
            let delta = sq[j..].iter().zip(&sq).map(|(a, b)| a * b).sum::<f64>() / (ss * ss);
            (2.0 * (qf - j as f64) / qf).powi(2) * delta
        })
        .sum();

    Ok(VarianceRatio {
        vr,
        z: (vr - 1.0) / phi.sqrt(),
        z_robust: (vr - 1.0) / theta.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RETURNS: [f64; 16] = [
        0.012, -0.007, 0.004, 0.015, -0.011, 0.003, -0.002, 0.009, -0.014, 0.006, 0.001, -0.005,
        0.013, -0.008, 0.002, 0.007,
    ];

    // (q, vr, z, z_robust) from Python `arch.unitroot.VarianceRatio`
    const REFERENCE: [(usize, f64, f64, f64); 2] = [
        (2, 0.414845134, -2.340619463, -2.468044888),
        (4, 0.370074869, -1.346836582, -1.576803609),
    ];

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn matches_reference() {
        for (q, vr, z, z_robust) in REFERENCE {
            let r = variance_ratio(&RETURNS, q).unwrap();
            assert!(
                close(r.vr, vr) && close(r.z, z) && close(r.z_robust, z_robust),
                "{r:?}"
            );
        }
    }

    #[test]
    fn direction() {
        let reverting: Vec<f64> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.2 })
            .collect();
        let persistent: Vec<f64> = (0..64)
            .map(|i| if i / 8 % 2 == 0 { 1.0 } else { -1.2 })
            .collect();
        assert!(variance_ratio(&reverting, 2).unwrap().vr < 0.5);
        assert!(variance_ratio(&persistent, 2).unwrap().vr > 1.5);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(variance_ratio(&RETURNS, 1).is_err());
        assert!(variance_ratio(&RETURNS, 16).is_err());
        assert!(variance_ratio(&[0.5; 16], 2).is_err());
        assert!(variance_ratio(&[f64::NAN; 16], 2).is_err());
    }
}
