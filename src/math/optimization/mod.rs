use ndarray::{Array1, Array2};

/// Minimize ½xᵀQx − cᵀx subject to 1ᵀx = 1, and x ≥ 0 unless `short`,
/// by projected gradient starting from `x`.
pub fn minimize_quadratic(
    q: &Array2<f64>,
    c: &Array1<f64>,
    mut x: Array1<f64>,
    short: bool,
) -> Array1<f64> {
    let step = 1.0 / q.iter().map(|v| v * v).sum::<f64>().sqrt();
    for _ in 0..100_000 {
        let v = &x - &((q.dot(&x) - c) * step);
        let next = if short {
            let shift = (v.sum() - 1.0) / v.len() as f64;
            v.mapv(|w| w - shift)
        } else {
            project_simplex(&v)
        };
        let moved = (&next - &x).iter().fold(0.0, |m: f64, d| m.max(d.abs()));
        x = next;
        if moved < 1e-13 {
            break;
        }
    }
    x
}

/// Euclidean projection onto {x ≥ 0, 1ᵀx = 1}
fn project_simplex(v: &Array1<f64>) -> Array1<f64> {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let (mut sum, mut shift) = (0.0, 0.0);
    for (k, &s) in sorted.iter().enumerate() {
        sum += s;
        let t = (sum - 1.0) / (k + 1) as f64;
        if s > t {
            shift = t;
        }
    }
    v.mapv(|w| (w - shift).max(0.0))
}
