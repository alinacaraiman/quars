#![allow(dead_code)]
/// Uniform noise in [-0.5, 0.5) from a fixed linear congruential generator
pub fn noise(n: usize) -> Vec<f64> {
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

/// r_t = phi * r_{t-1} + noise
pub fn ar1(phi: f64, n: usize) -> Vec<f64> {
    let mut r = 0.0;
    noise(n)
        .iter()
        .map(|e| {
            r = phi * r + e;
            r
        })
        .collect()
}
