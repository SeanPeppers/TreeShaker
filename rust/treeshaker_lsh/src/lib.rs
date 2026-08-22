//! SimHash-based locality-sensitive hashing for near-linear-in-p
//! near-duplicate / high-correlation feature screening.
//!
//! Exact pairwise correlation screening (as done in
//! `treeshaker.leakage_detection.LeakageDetector._check_near_duplicates`)
//! is O(p^2) over the feature count `p`, which is fine for narrow tabular
//! data but becomes a bottleneck on wide data (thousands of features --
//! genomics, embeddings, etc). This crate provides an *approximate*,
//! opt-in alternative: each feature column is projected onto a small
//! number of random hyperplanes (SimHash), producing a short bit
//! signature per feature. Two highly-correlated (or near-duplicate)
//! columns tend to land on the same side of most random hyperplanes, so
//! they collide in signature space far more often than unrelated columns.
//! Bucketing features by signature turns the all-pairs comparison into a
//! comparison only within (small) buckets, making the expected work
//! close to O(p) for typical wide, sparse-collision data, instead of
//! O(p^2).
//!
//! This module deliberately exposes a single, narrow entry point
//! (`simhash_screen`) so the Python side has a minimal, stable surface to
//! depend on. All correlation candidates returned are *approximate* --
//! callers are expected to treat them as candidates for further
//! inspection, not as a drop-in replacement guaranteed to find every
//! near-duplicate the exact path would find.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Compute a SimHash signature for a single feature column.
///
/// `column` is the (already standardized -- mean 0) feature values, and
/// `hyperplanes` is a `num_hashes x n_rows` matrix of random unit-ish
/// projection vectors shared across all features so that signatures are
/// comparable to one another. Each bit of the signature is set
/// according to the sign of the dot product of `column` with one
/// hyperplane row.
fn simhash_signature(column: &[f64], hyperplanes: &[Vec<f64>]) -> u64 {
    let mut signature: u64 = 0;
    for (bit, plane) in hyperplanes.iter().enumerate() {
        let dot: f64 = column.iter().zip(plane.iter()).map(|(a, b)| a * b).sum();
        if dot >= 0.0 {
            signature |= 1u64 << bit;
        }
    }
    signature
}

/// Pearson correlation between two equal-length numeric slices.
///
/// Returns `None` if either slice has zero variance (correlation is
/// undefined), mirroring how the Python exact path skips NaN
/// correlations.
fn pearson_corr(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len() as f64;
    if n == 0.0 {
        return None;
    }
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return None;
    }
    Some(cov / (var_a.sqrt() * var_b.sqrt()))
}

/// Signatures are packed into a `u64`, so `num_hashes` must be in `1..=64`
/// -- otherwise the bit shift in `simhash_signature` would overflow.
fn check_num_hashes(num_hashes: usize) -> PyResult<()> {
    if num_hashes == 0 || num_hashes > 64 {
        return Err(PyValueError::new_err(format!(
            "num_hashes must be between 1 and 64 (got {num_hashes}); \
             signatures are packed into a u64"
        )));
    }
    Ok(())
}

/// Screen a wide feature matrix for candidate near-duplicate / highly
/// correlated feature pairs using SimHash LSH, in near-linear time in
/// the number of features.
///
/// Args (from Python):
///     features: a 2D float64 array of shape (n_features, n_rows) --
///         each row is one feature's values across all samples.
///     num_hashes: number of random hyperplanes used to build each
///         feature's SimHash signature. More hashes means finer-grained
///         bucketing (fewer false-positive candidate pairs) at the cost
///         of more buckets that may miss true near-duplicates that
///         happen to disagree on a hyperplane. 16-32 is a reasonable
///         range for thousands of features.
///     seed: RNG seed for the random hyperplanes, for reproducibility.
///
/// Returns:
///     A list of (i, j, correlation) tuples for column-index pairs that
///     hashed into the same SimHash bucket and whose exact Pearson
///     correlation was computed as part of the bucket-local comparison.
///     `i < j` always. This is the *candidate* set the LSH step
///     produced -- correlations are then exact for any pair that
///     collided, so precision within reported pairs is exact; the
///     approximation only affects which true near-duplicate pairs may be
///     *missed* because they did not collide in any bucket.
#[pyfunction]
#[pyo3(signature = (features, num_hashes=24, seed=42))]
fn simhash_screen(
    features: PyReadonlyArray2<f64>,
    num_hashes: usize,
    seed: u64,
) -> PyResult<Vec<(usize, usize, f64)>> {
    check_num_hashes(num_hashes)?;
    let arr = features.as_array();
    let n_features = arr.shape()[0];
    let n_rows = arr.shape()[1];

    if n_features == 0 || n_rows == 0 {
        return Ok(Vec::new());
    }

    // Generate shared random hyperplanes (num_hashes x n_rows), reused
    // for every feature so signatures are directly comparable.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    let hyperplanes: Vec<Vec<f64>> = (0..num_hashes)
        .map(|_| (0..n_rows).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    // Mean-center each column (signature only depends on direction, but
    // centering matches what a correlation-screening signature should
    // look at -- co-movement around the mean, not absolute level).
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    let mut signatures: Vec<u64> = Vec::with_capacity(n_features);
    for i in 0..n_features {
        let row = arr.row(i);
        let mean = row.iter().sum::<f64>() / (n_rows as f64);
        let centered: Vec<f64> = row.iter().map(|v| v - mean).collect();
        let sig = simhash_signature(&centered, &hyperplanes);
        columns.push(centered);
        signatures.push(sig);
    }

    // Bucket feature indices by signature.
    use std::collections::HashMap;
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
    for (idx, &sig) in signatures.iter().enumerate() {
        buckets.entry(sig).or_default().push(idx);
    }

    // Within each bucket, compute exact correlation for all pairs --
    // buckets are expected to be small relative to n_features, so this
    // stays close to O(p) overall rather than O(p^2).
    let mut results = Vec::new();
    for bucket in buckets.values() {
        if bucket.len() < 2 {
            continue;
        }
        for a in 0..bucket.len() {
            for b in (a + 1)..bucket.len() {
                let i = bucket[a];
                let j = bucket[b];
                if let Some(corr) = pearson_corr(&columns[i], &columns[j]) {
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    results.push((lo, hi, corr));
                }
            }
        }
    }

    Ok(results)
}

/// Compute a SimHash signature (as a u64 bit pattern) for a single
/// 1D feature vector. Exposed mainly for testing/debugging the
/// signature construction independent of the bucketing/screening logic.
#[pyfunction]
#[pyo3(signature = (column, num_hashes=24, seed=42))]
fn simhash_signature_py(
    column: PyReadonlyArray1<f64>,
    num_hashes: usize,
    seed: u64,
) -> PyResult<u64> {
    check_num_hashes(num_hashes)?;
    let arr = column.as_array();
    let n_rows = arr.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    let hyperplanes: Vec<Vec<f64>> = (0..num_hashes)
        .map(|_| (0..n_rows).map(|_| normal.sample(&mut rng)).collect())
        .collect();
    let mean = arr.iter().sum::<f64>() / (n_rows as f64);
    let centered: Vec<f64> = arr.iter().map(|v| v - mean).collect();
    Ok(simhash_signature(&centered, &hyperplanes))
}

#[pymodule]
fn treeshaker_lsh(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simhash_screen, m)?)?;
    m.add_function(wrap_pyfunction!(simhash_signature_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_corr_perfect_positive() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0];
        let corr = pearson_corr(&a, &b).unwrap();
        assert!((corr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_corr_constant_is_none() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(pearson_corr(&a, &b).is_none());
    }

    #[test]
    fn simhash_signature_identical_columns_match() {
        let hyperplanes = vec![vec![1.0, -1.0, 0.5], vec![0.2, 0.3, -0.4]];
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(simhash_signature(&a, &hyperplanes), simhash_signature(&b, &hyperplanes));
    }

    #[test]
    fn check_num_hashes_rejects_out_of_range() {
        assert!(check_num_hashes(0).is_err());
        assert!(check_num_hashes(65).is_err());
        assert!(check_num_hashes(1).is_ok());
        assert!(check_num_hashes(64).is_ok());
    }
}
