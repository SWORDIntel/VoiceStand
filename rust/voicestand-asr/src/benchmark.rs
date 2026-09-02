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

/// Case-insensitive word error rate using Levenshtein edit distance.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> Option<f64> {
    let reference = words(reference);
    if reference.is_empty() {
        return None;
    }
    let hypothesis = words(hypothesis);
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    for (row, expected) in reference.iter().enumerate() {
        let mut current = Vec::with_capacity(hypothesis.len() + 1);
        current.push(row + 1);
        for (column, actual) in hypothesis.iter().enumerate() {
            current.push(
                (previous[column + 1] + 1)
                    .min(current[column] + 1)
                    .min(previous[column] + usize::from(expected != actual)),
            );
        }
        previous = current;
    }
    Some(previous[hypothesis.len()] as f64 / reference.len() as f64)
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect()
        })
        .filter(|word: &String| !word.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{percentile, word_error_rate};

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

    #[test]
    fn calculates_normalized_word_error_rate() {
        assert_eq!(
            word_error_rate("Ask not, what you can do", "ask what you can do"),
            Some(1.0 / 6.0)
        );
        assert_eq!(word_error_rate("", "anything"), None);
    }
}
