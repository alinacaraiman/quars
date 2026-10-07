pub mod correlation;
pub mod nonlinear;
pub mod optimization;
pub mod scaling;
pub mod variance_ratio;

/// Standard normal distribution function
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal density
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// P(|Z| > |z|) for a standard normal Z
pub(crate) fn two_sided_p(z: f64) -> f64 {
    2.0 * normal_cdf(-z.abs())
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
