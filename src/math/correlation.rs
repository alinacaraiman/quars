use crate::Error;
use ndarray::{Array1, Array2};

/// Correlation matrix of a covariance matrix
pub fn correlation(cov: &Array2<f64>) -> Result<Array2<f64>, Error> {
    let std = cov.diag().mapv(f64::sqrt);
    if std.iter().any(|s| !(*s > 0.0)) {
        return Err("Variances must be positive.".into());
    }
    Ok(Array2::from_shape_fn(cov.dim(), |(i, j)| {
        cov[[i, j]] / (std[i] * std[j])
    }))
}

/// Random-matrix cleaning by eigenvalue clipping (Laloux et al., 1999).
/// Eigenvalues of `corr` up to the Marchenko-Pastur edge (1 + √(N/T))², for N assets
/// and T = `n_samples`, count as noise and are replaced by their average.
pub fn clean_correlation(corr: &Array2<f64>, n_samples: usize) -> Result<Array2<f64>, Error> {
    let n = corr.nrows();
    let edge = (1.0 + (n as f64 / n_samples as f64).sqrt()).powi(2);
    let (mut values, vectors) = symmetric_eigen(corr);
    let noise: Vec<usize> = (0..n).filter(|&i| values[i] <= edge).collect();
    let average = noise.iter().map(|&i| values[i]).sum::<f64>() / noise.len() as f64;
    for i in noise {
        values[i] = average;
    }
    correlation(&vectors.dot(&Array2::from_diag(&values)).dot(&vectors.t()))
}

/// Minimum spanning tree of the correlation distances √(2(1 − ρ)) (Mantegna, 1999),
/// as N − 1 edges (i, j, distance)
pub fn minimum_spanning_tree(corr: &Array2<f64>) -> Vec<(usize, usize, f64)> {
    let n = corr.nrows();
    let distance = |i: usize, j: usize| (2.0 * (1.0 - corr[[i, j]])).max(0.0).sqrt();
    // Prim's algorithm from asset 0: (distance to the tree, nearest tree asset)
    let mut nearest: Vec<(f64, usize)> = (0..n).map(|j| (distance(0, j), 0)).collect();
    let mut in_tree = vec![false; n];
    let mut edges = Vec::new();
    let mut last = 0;
    for _ in 1..n {
        in_tree[last] = true;
        for k in (0..n).filter(|&k| !in_tree[k]) {
            if distance(last, k) < nearest[k].0 {
                nearest[k] = (distance(last, k), last);
            }
        }
        let Some(next) = (0..n)
            .filter(|&k| !in_tree[k])
            .min_by(|&a, &b| nearest[a].0.total_cmp(&nearest[b].0))
        else {
            break;
        };
        edges.push((nearest[next].1, next, nearest[next].0));
        last = next;
    }
    edges
}

/// Eigenvalues and eigenvectors (columns) of a symmetric matrix by cyclic Jacobi rotations
fn symmetric_eigen(matrix: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    fn rotate_columns(m: &mut Array2<f64>, p: usize, q: usize, c: f64, s: f64) {
        for k in 0..m.nrows() {
            let (mp, mq) = (m[[k, p]], m[[k, q]]);
            m[[k, p]] = c * mp - s * mq;
            m[[k, q]] = s * mp + c * mq;
        }
    }
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut vectors = Array2::<f64>::eye(n);
    for _ in 0..100 {
        let off_diagonal: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[[p, q]].powi(2))
            .sum();
        if off_diagonal < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[[p, q]] == 0.0 {
                    continue;
                }
                let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                rotate_columns(&mut a, p, q, c, s);
                a.swap_axes(0, 1);
                rotate_columns(&mut a, p, q, c, s);
                a.swap_axes(0, 1);
                rotate_columns(&mut vectors, p, q, c, s);
            }
        }
    }
    (a.diag().to_owned(), vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sample correlation of 6 simulated series with 60 observations, three sharing a factor
    const CORR: [[f64; 6]; 6] = [
        [
            1.0,
            0.444622088,
            0.528842242,
            -0.32385771,
            -0.008611822,
            0.207691699,
        ],
        [
            0.444622088,
            1.0,
            0.282151532,
            -0.23810158,
            0.136353523,
            0.211231027,
        ],
        [
            0.528842242,
            0.282151532,
            1.0,
            -0.205740209,
            -0.090152538,
            0.206340663,
        ],
        [
            -0.32385771,
            -0.23810158,
            -0.205740209,
            1.0,
            0.078762944,
            -0.14922208,
        ],
        [
            -0.008611822,
            0.136353523,
            -0.090152538,
            0.078762944,
            1.0,
            0.0019484,
        ],
        [
            0.207691699,
            0.211231027,
            0.206340663,
            -0.14922208,
            0.0019484,
            1.0,
        ],
    ];
    // Eigenvalue clipping of `CORR` with numpy
    const CLEANED: [[f64; 6]; 6] = [
        [
            1.0,
            0.318284803,
            0.330331048,
            -0.272049662,
            -0.010194505,
            0.236382371,
        ],
        [
            0.318284803,
            1.0,
            0.290922579,
            -0.239594158,
            -0.008978301,
            0.208181972,
        ],
        [
            0.330331048,
            0.290922579,
            1.0,
            -0.248662168,
            -0.009318106,
            0.216061114,
        ],
        [
            -0.272049662,
            -0.239594158,
            -0.248662168,
            1.0,
            0.007674082,
            -0.177940745,
        ],
        [
            -0.010194505,
            -0.008978301,
            -0.009318106,
            0.007674082,
            1.0,
            -0.006667966,
        ],
        [
            0.236382371,
            0.208181972,
            0.216061114,
            -0.177940745,
            -0.006667966,
            1.0,
        ],
    ];

    fn matrix(rows: [[f64; 6]; 6]) -> Array2<f64> {
        Array2::from_shape_fn((6, 6), |(i, j)| rows[i][j])
    }

    fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert!(
            (actual - expected).iter().all(|d| d.abs() < 1e-6),
            "{actual:?}"
        );
    }

    #[test]
    fn correlation_of_covariance() {
        let std = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let cov = Array2::from_shape_fn((6, 6), |(i, j)| CORR[i][j] * std[i] * std[j]);
        assert_close(&correlation(&cov).unwrap(), &matrix(CORR));
        assert!(correlation(&Array2::zeros((2, 2))).is_err());
    }

    #[test]
    fn cleaning_matches_reference() {
        assert_close(
            &clean_correlation(&matrix(CORR), 60).unwrap(),
            &matrix(CLEANED),
        );
    }

    // Edges and total length from scipy `minimum_spanning_tree`
    #[test]
    fn spanning_tree_matches_reference() {
        let mut edges: Vec<(usize, usize)> = minimum_spanning_tree(&matrix(CORR))
            .iter()
            .map(|&(i, j, _)| (i.min(j), i.max(j)))
            .collect();
        edges.sort();
        assert_eq!(edges, [(0, 1), (0, 2), (1, 4), (1, 5), (3, 4)]);
        let total: f64 = minimum_spanning_tree(&matrix(CORR))
            .iter()
            .map(|e| e.2)
            .sum();
        assert!((total - 5.952296997).abs() < 1e-6);
    }
}
