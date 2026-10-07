use std::error::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceRatio {
    pub vr: f64,
    /// Test statistic under homoskedasticity
    pub z: f64,
    /// Heteroskedasticity-robust test statistic
    pub z_robust: f64,
    /// Two-sided p-values of `z` and `z_robust`
    pub p: f64,
    pub p_robust: f64,
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

    let (z, z_robust) = ((vr - 1.0) / phi.sqrt(), (vr - 1.0) / theta.sqrt());
    Ok(VarianceRatio {
        vr,
        z,
        z_robust,
        p: two_sided_p(z),
        p_robust: two_sided_p(z_robust),
    })
}

/// P(|Z| > |z|) for a standard normal Z, erfc by Abramowitz-Stegun 7.1.26 (error < 1.5e-7)
fn two_sided_p(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RETURNS: [f64; 16] = [
        0.012, -0.007, 0.004, 0.015, -0.011, 0.003, -0.002, 0.009, -0.014, 0.006, 0.001, -0.005,
        0.013, -0.008, 0.002, 0.007,
    ];

    // (q, vr, z, z_robust, p, p_robust) from Python `arch.unitroot.VarianceRatio`
    const REFERENCE: [(usize, [f64; 5]); 2] = [
        (
            2,
            [
                0.414845134,
                -2.340619463,
                -2.468044888,
                0.019251778,
                0.013585328,
            ],
        ),
        (
            4,
            [
                0.370074869,
                -1.346836582,
                -1.576803609,
                0.178032870,
                0.114840721,
            ],
        ),
    ];

    // r_t = phi * r_{t-1} + e_t
    fn ar1(phi: f64) -> Vec<f64> {
        let mut r = 0.0;
        crate::math::test_noise(1024)
            .iter()
            .map(|e| {
                r = phi * r + e;
                r
            })
            .collect()
    }

    #[test]
    fn matches_reference() {
        for (q, expected) in REFERENCE {
            let r = variance_ratio(&RETURNS, q).unwrap();
            let actual = [r.vr, r.z, r.z_robust, r.p, r.p_robust];
            for (a, e) in actual.iter().zip(expected) {
                assert!((a - e).abs() < 1e-6, "{r:?}");
            }
        }
    }

    #[test]
    fn iid_is_not_rejected() {
        for q in [2, 4] {
            let r = variance_ratio(&ar1(0.0), q).unwrap();
            assert!((r.vr - 1.0).abs() < 0.15 && r.p_robust > 0.05, "{r:?}");
        }
    }

    #[test]
    fn ar1_is_rejected() {
        let persistent = variance_ratio(&ar1(0.5), 2).unwrap();
        assert!(
            persistent.vr > 1.3 && persistent.p_robust < 0.01,
            "{persistent:?}"
        );
        let reverting = variance_ratio(&ar1(-0.5), 2).unwrap();
        assert!(
            reverting.vr < 0.7 && reverting.p_robust < 0.01,
            "{reverting:?}"
        );
    }

    #[test]
    fn rejects_bad_input() {
        assert!(variance_ratio(&RETURNS, 1).is_err());
        assert!(variance_ratio(&RETURNS, 16).is_err());
        assert!(variance_ratio(&[0.5; 16], 2).is_err());
        assert!(variance_ratio(&[f64::NAN; 16], 2).is_err());
    }
}
