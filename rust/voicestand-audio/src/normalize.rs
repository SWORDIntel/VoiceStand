/// Convert interleaved device audio to mono at the target sample rate.
pub fn normalize_audio(
    input: &[f32],
    source_rate: u32,
    source_channels: u16,
    target_rate: u32,
) -> Vec<f32> {
    if input.is_empty() || source_rate == 0 || source_channels == 0 || target_rate == 0 {
        return Vec::new();
    }

    let channels = usize::from(source_channels);
    let mono: Vec<f32> = input
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    if mono.is_empty() || source_rate == target_rate {
        return mono;
    }

    let output_len =
        ((mono.len() as u64 * u64::from(target_rate)) / u64::from(source_rate)).max(1) as usize;
    let scale = source_rate as f64 / target_rate as f64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * scale;
            let left = position.floor() as usize;
            let right = (left + 1).min(mono.len() - 1);
            let fraction = (position - left as f64) as f32;
            mono[left.min(mono.len() - 1)] * (1.0 - fraction) + mono[right] * fraction
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmixes_stereo() {
        assert_eq!(
            normalize_audio(&[1.0, -1.0, 0.5, 0.5], 16_000, 2, 16_000),
            vec![0.0, 0.5]
        );
    }

    #[test]
    fn resamples_to_target_rate() {
        let output = normalize_audio(&[0.0, 1.0, 0.0, -1.0], 8_000, 1, 16_000);
        assert_eq!(output.len(), 8);
        assert_eq!(&output[..4], &[0.0, 0.5, 1.0, 0.5]);
    }
}
