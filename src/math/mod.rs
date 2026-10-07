pub mod correlation;
pub mod optimization;
pub mod scaling;
pub mod variance_ratio;

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
