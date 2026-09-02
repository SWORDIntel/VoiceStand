use std::path::Path;

use hound::{SampleFormat, WavReader};
use voicestand_types::{Result, VoiceStandError};

/// Read the fixed corpus format used by production ASR benchmarks.
///
/// Files must contain 16 kHz mono or stereo PCM. Stereo input is downmixed.
pub fn read_wav_16khz_mono(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let mut reader = WavReader::open(path).map_err(|error| {
        VoiceStandError::audio(format!("could not open WAV {}: {error}", path.display()))
    })?;
    let spec = reader.spec();

    if spec.sample_rate != 16_000 {
        return Err(VoiceStandError::audio(format!(
            "WAV sample rate must be 16000 Hz, got {}",
            spec.sample_rate
        )));
    }
    if !(spec.channels == 1 || spec.channels == 2) {
        return Err(VoiceStandError::audio(format!(
            "WAV must be mono or stereo, got {} channels",
            spec.channels
        )));
    }

    let interleaved = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|value| value as f32 / i16::MAX as f32)
                    .map_err(|error| VoiceStandError::audio(format!("invalid WAV sample: {error}")))
            })
            .collect::<Result<Vec<_>>>()?,
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|sample| {
                sample
                    .map_err(|error| VoiceStandError::audio(format!("invalid WAV sample: {error}")))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(VoiceStandError::audio(format!(
                "unsupported WAV encoding: {:?}/{}-bit",
                spec.sample_format, spec.bits_per_sample
            )))
        }
    };

    if spec.channels == 1 {
        return Ok(interleaved);
    }

    Ok(interleaved
        .chunks_exact(2)
        .map(|frame| (frame[0] + frame[1]) * 0.5)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};

    #[test]
    fn reads_and_downmixes_fixed_corpus_wav() {
        let path =
            std::env::temp_dir().join(format!("voicestand-asr-wav-{}.wav", std::process::id()));
        let spec = WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).expect("create WAV fixture");
        writer.write_sample(i16::MAX).expect("write left");
        writer.write_sample(0_i16).expect("write right");
        writer.finalize().expect("finalize WAV fixture");

        let samples = read_wav_16khz_mono(&path).expect("read WAV fixture");
        assert_eq!(samples, vec![0.5]);

        std::fs::remove_file(path).expect("remove WAV fixture");
    }
}
