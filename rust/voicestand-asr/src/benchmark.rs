//! Small helpers shared by benchmark frontends.

/// Nearest-rank percentile for latency samples.
pub fn percentile(samples: &[f64], percentile: f64) -> Option<f64> {
    if samples.is_empty() || !percentile.is_finite() || !(0.0..=100.0).contains(&percentile) {
        return None;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let rank = ((percentile / 100.0) * ordered.len() as f64).ceil() as usize;
    ordered
        .get(rank.saturating_sub(1).min(ordered.len() - 1))
        .copied()
}

#[cfg(test)]
mod tests {
    use super::percentile;

    #[test]
    fn calculates_nearest_rank_percentiles() {
        let samples = [10.0, 50.0, 20.0, 40.0, 30.0];
        assert_eq!(percentile(&samples, 50.0), Some(30.0));
        assert_eq!(percentile(&samples, 95.0), Some(50.0));
        assert_eq!(percentile(&samples, 0.0), Some(10.0));
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(percentile(&[], 50.0), None);
        assert_eq!(percentile(&[1.0], 101.0), None);
        assert_eq!(percentile(&[1.0], f64::NAN), None);
    }
}
