use voicestand_core::{TranscriptionResult, Result};
use std::collections::HashMap;
use regex::Regex;

/// Post-processing utilities for transcription results
pub struct TranscriptionPostProcessor {
    punctuation_rules: Vec<PunctuationRule>,
    abbreviations: HashMap<String, String>,
    confidence_threshold: f32,
}

impl TranscriptionPostProcessor {
    pub fn new() -> Self {
        Self {
            punctuation_rules: Self::default_punctuation_rules(),
            abbreviations: Self::default_abbreviations(),
            confidence_threshold: 0.5,
        }
    }

    /// Process a transcription result with all enhancements
    pub fn process(&self, mut result: TranscriptionResult) -> Result<TranscriptionResult> {
        // Skip processing if confidence is too low
        if result.confidence < self.confidence_threshold {
            return Ok(result);
        }

        // Apply processing steps
        result.text = self.normalize_text(&result.text);
        result.text = self.expand_abbreviations(&result.text);
        result.text = self.add_punctuation(&result.text);
        result.text = self.capitalize_sentences(&result.text);
        result.text = self.clean_whitespace(&result.text);

        Ok(result)
    }

    /// Normalize text (lowercase, remove extra spaces)
    fn normalize_text(&self, text: &str) -> String {
        text.trim().to_lowercase()
    }

    /// Expand common abbreviations
    fn expand_abbreviations(&self, text: &str) -> String {
        let mut result = text.to_string();

        for (abbrev, expansion) in &self.abbreviations {
            let pattern = format!(r"\b{}\b", regex::escape(abbrev));
            if let Ok(re) = Regex::new(&pattern) {
                result = re.replace_all(&result, expansion).to_string();
            }
        }

        result
    }

    /// Add punctuation based on patterns
    fn add_punctuation(&self, text: &str) -> String {
        let mut result = text.to_string();

        for rule in &self.punctuation_rules {
            if let Ok(re) = Regex::new(&rule.pattern) {
                result = re.replace_all(&result, &rule.replacement).to_string();
            }
        }

        result
    }

    /// Capitalize sentences
    fn capitalize_sentences(&self, text: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = true;

        for ch in text.chars() {
            if ch.is_alphabetic() && capitalize_next {
                result.push(ch.to_uppercase().next().unwrap_or(ch));
                capitalize_next = false;
            } else {
                result.push(ch);
                if ch == '.' || ch == '!' || ch == '?' {
                    capitalize_next = true;
                }
            }
        }

        result
    }

    /// Clean up whitespace
    fn clean_whitespace(&self, text: &str) -> String {
        // Remove multiple spaces
        let re = Regex::new(r"\s+").expect("Regex should compile successfully");
        let text = re.replace_all(text, " ");

        // Fix spacing around punctuation
        let fixes = vec![
            (r" +([,.!?:;])", "$1"),
            (r"([,.!?:;]) +", "$1 "),
            (r" +(['\"])", "$1"),
            (r"(['\"] +)", "$1"),
        ];

        let mut result = text.to_string();
        for (pattern, replacement) in fixes {
            if let Ok(re) = Regex::new(pattern) {
                result = re.replace_all(&result, replacement).to_string();
            }
        }

        result.trim().to_string()
    }

    /// Default punctuation rules
    fn default_punctuation_rules() -> Vec<PunctuationRule> {
        vec![
            // Question patterns
            PunctuationRule {
                pattern: r"\b(what|where|when|why|who|how|is|are|do|does|did|can|could|would|will)\b.*".to_string(),
                replacement: "$0?".to_string(),
                confidence: 0.8,
            },
            // Period at end of statements
            PunctuationRule {
                pattern: r"([a-z])\s*$".to_string(),
                replacement: "$1.".to_string(),
                confidence: 0.9,
            },
            // Comma before "and", "but", "or" in longer sentences
            PunctuationRule {
                pattern: r"\b(and|but|or)\b".to_string(),
                replacement: ", $1".to_string(),
                confidence: 0.6,
            },
            // Exclamation for exclamatory words
            PunctuationRule {
                pattern: r"\b(wow|amazing|incredible|fantastic|terrible|awful)\b.*".to_string(),
                replacement: "$0!".to_string(),
                confidence: 0.7,
            },
        ]
    }

    /// Default abbreviations
    fn default_abbreviations() -> HashMap<String, String> {
        let mut abbrevs = HashMap::new();

        // Common abbreviations
        abbrevs.insert("dont".to_string(), "don't".to_string());
        abbrevs.insert("cant".to_string(), "can't".to_string());
        abbrevs.insert("wont".to_string(), "won't".to_string());
        abbrevs.insert("isnt".to_string(), "isn't".to_string());
        abbrevs.insert("arent".to_string(), "aren't".to_string());
        abbrevs.insert("wasnt".to_string(), "wasn't".to_string());
        abbrevs.insert("werent".to_string(), "weren't".to_string());
        abbrevs.insert("hasnt".to_string(), "hasn't".to_string());
        abbrevs.insert("havent".to_string(), "haven't".to_string());
        abbrevs.insert("hadnt".to_string(), "hadn't".to_string());
        abbrevs.insert("wouldnt".to_string(), "wouldn't".to_string());
        abbrevs.insert("couldnt".to_string(), "couldn't".to_string());
        abbrevs.insert("shouldnt".to_string(), "shouldn't".to_string());

        // Numbers
        abbrevs.insert("one".to_string(), "1".to_string());
        abbrevs.insert("two".to_string(), "2".to_string());
        abbrevs.insert("three".to_string(), "3".to_string());
        abbrevs.insert("four".to_string(), "4".to_string());
        abbrevs.insert("five".to_string(), "5".to_string());
        abbrevs.insert("six".to_string(), "6".to_string());
        abbrevs.insert("seven".to_string(), "7".to_string());
        abbrevs.insert("eight".to_string(), "8".to_string());
        abbrevs.insert("nine".to_string(), "9".to_string());
        abbrevs.insert("ten".to_string(), "10".to_string());

        // Tech terms
        abbrevs.insert("cpu".to_string(), "CPU".to_string());
        abbrevs.insert("gpu".to_string(), "GPU".to_string());
        abbrevs.insert("ram".to_string(), "RAM".to_string());
        abbrevs.insert("usb".to_string(), "USB".to_string());
        abbrevs.insert("api".to_string(), "API".to_string());
        abbrevs.insert("url".to_string(), "URL".to_string());
        abbrevs.insert("http".to_string(), "HTTP".to_string());
        abbrevs.insert("https".to_string(), "HTTPS".to_string());

        abbrevs
    }

    /// Update confidence threshold
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Add custom abbreviation
    pub fn add_abbreviation(&mut self, abbrev: String, expansion: String) {
        self.abbreviations.insert(abbrev, expansion);
    }

    /// Add custom punctuation rule
    pub fn add_punctuation_rule(&mut self, rule: PunctuationRule) {
        self.punctuation_rules.push(rule);
    }
}

/// Punctuation rule for pattern-based punctuation restoration
#[derive(Debug, Clone)]
pub struct PunctuationRule {
    pub pattern: String,
    pub replacement: String,
    pub confidence: f32,
}

impl Default for TranscriptionPostProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Speaker diarization utilities
pub struct SpeakerDiarization {
    speaker_profiles: HashMap<String, SpeakerProfile>,
    current_speaker: Option<String>,
}

impl SpeakerDiarization {
    pub fn new() -> Self {
        Self {
            speaker_profiles: HashMap::new(),
            current_speaker: None,
        }
    }

    /// Identify speaker from audio features (simplified)
    pub fn identify_speaker(&mut self, audio_features: &[f32]) -> Option<String> {
        let mfcc = self.extract_mfcc_features(audio_features);

        // Find best matching speaker
        let mut best_match = None;
        let mut best_similarity = 0.0;

        for (speaker_id, profile) in &self.speaker_profiles {
            let similarity = self.calculate_similarity(&mfcc, &profile.features);
            if similarity > best_similarity && similarity > 0.7 {
                best_similarity = similarity;
                best_match = Some(speaker_id.clone());
            }
        }

        if let Some(speaker) = best_match {
            self.current_speaker = Some(speaker.clone());
            Some(speaker)
        } else {
            // Unknown speaker
            let new_speaker = format!("Speaker_{}", self.speaker_profiles.len() + 1);
            self.add_speaker_profile(new_speaker.clone(), mfcc);
            self.current_speaker = Some(new_speaker.clone());
            Some(new_speaker)
        }
    }

    /// Add a new speaker profile
    pub fn add_speaker_profile(&mut self, speaker_id: String, features: Vec<f32>) {
        self.speaker_profiles.insert(speaker_id, SpeakerProfile {
            features,
            sample_count: 1,
        });
    }

    /// Update existing speaker profile
    pub fn update_speaker_profile(&mut self, speaker_id: &str, new_features: Vec<f32>) {
        if let Some(profile) = self.speaker_profiles.get_mut(speaker_id) {
            // Simple moving average update
            let alpha = 0.1; // Learning rate
            for (i, &new_val) in new_features.iter().enumerate() {
                if i < profile.features.len() {
                    profile.features[i] = profile.features[i] * (1.0 - alpha) + new_val * alpha;
                }
            }
            profile.sample_count += 1;
        }
    }

    /// Extract MFCC features (simplified)
    fn extract_mfcc_features(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified MFCC extraction - in real implementation would use proper DSP
        let mut features = Vec::with_capacity(13); // 13 MFCC coefficients

        // Calculate basic spectral features
        let energy = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        features.push(energy.sqrt());

        // Zero crossing rate
        let mut crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        features.push(crossings as f32 / audio.len() as f32);

        // Spectral centroid (simplified)
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        for (i, &sample) in audio.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += i as f32 * magnitude;
            magnitude_sum += magnitude;
        }
        let centroid = if magnitude_sum > 0.0 { weighted_sum / magnitude_sum } else { 0.0 };
        features.push(centroid);

        // Pad to 13 coefficients with zeros or derived features
        while features.len() < 13 {
            features.push(0.0);
        }

        features
    }

    /// Calculate similarity between feature vectors
    fn calculate_similarity(&self, features1: &[f32], features2: &[f32]) -> f32 {
        if features1.len() != features2.len() {
            return 0.0;
        }

        // Cosine similarity
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for i in 0..features1.len() {
            dot_product += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        } else {
            0.0
        }
    }

    /// Get current speaker
    pub fn current_speaker(&self) -> Option<&str> {
        self.current_speaker.as_deref()
    }

    /// Get all known speakers
    pub fn get_speakers(&self) -> Vec<String> {
        self.speaker_profiles.keys().cloned().collect()
    }
}

/// Speaker profile for diarization
#[derive(Debug, Clone)]
struct SpeakerProfile {
    features: Vec<f32>,
    sample_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_normalization() {
        let processor = TranscriptionPostProcessor::new();
        let text = "  HELLO WORLD  ";
        let normalized = processor.normalize_text(text);
        assert_eq!(normalized, "hello world");
    }

    #[test]
    fn test_abbreviation_expansion() {
        let processor = TranscriptionPostProcessor::new();
        let text = "dont do that";
        let expanded = processor.expand_abbreviations(text);
        assert_eq!(expanded, "don't do that");
    }

    #[test]
    fn test_capitalization() {
        let processor = TranscriptionPostProcessor::new();
        let text = "hello world. how are you?";
        let capitalized = processor.capitalize_sentences(text);
        assert_eq!(capitalized, "Hello world. How are you?");
    }

    #[test]
    fn test_whitespace_cleanup() {
        let processor = TranscriptionPostProcessor::new();
        let text = "hello  ,  world   .   how   are   you  ?";
        let cleaned = processor.clean_whitespace(text);
        assert_eq!(cleaned, "hello, world. how are you?");
    }

    #[test]
    fn test_full_processing() {
        let processor = TranscriptionPostProcessor::new();
        let mut result = TranscriptionResult::new(
            "hello world dont do that".to_string(),
            0.8,
            0.0,
            1.0,
            true,
        );

        let processed = processor.process(result).expect("Post-processing should succeed");
        assert_eq!(processed.text, "Hello world don't do that.");
    }

    #[test]
    fn test_speaker_diarization() {
        let mut diarization = SpeakerDiarization::new();
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let speaker = diarization.identify_speaker(&features);
        assert!(speaker.is_some());
        assert_eq!(diarization.get_speakers().len(), 1);
    }
}

// Add regex dependency to Cargo.toml