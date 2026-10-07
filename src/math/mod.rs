pub mod correlation;
pub mod nonlinear;
pub mod optimization;
pub mod scaling;
pub mod variance_ratio;

/// P(|Z| > |z|) for a standard normal Z, erfc by Abramowitz-Stegun 7.1.26 (error < 1.5e-7)
pub(crate) fn two_sided_p(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// Uniform noise in [-0.5, 0.5) from a fixed linear congruential generator
#[cfg(test)]
pub(crate) fn test_noise(n: usize) -> Vec<f64> {
    let mut state = 1u64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}
