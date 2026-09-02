#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StabilizedPartial {
    pub text: String,
    pub stable_prefix_bytes: usize,
}

#[derive(Default)]
pub struct TranscriptStabilizer {
    previous: String,
}

impl TranscriptStabilizer {
    pub fn update(&mut self, text: &str) -> Option<StabilizedPartial> {
        let text = text.trim();
        if text.is_empty() || text == self.previous {
            return None;
        }
        let stable_prefix_bytes = stable_word_prefix(&self.previous, text);
        self.previous.clear();
        self.previous.push_str(text);
        Some(StabilizedPartial {
            text: text.to_string(),
            stable_prefix_bytes,
        })
    }

    pub fn reset(&mut self) {
        self.previous.clear();
    }
}

fn stable_word_prefix(left: &str, right: &str) -> usize {
    let mut boundary = 0;
    for ((left_index, left_word), (right_index, right_word)) in left
        .split_word_bound_indices()
        .into_iter()
        .zip(right.split_word_bound_indices())
    {
        if left_word != right_word {
            break;
        }
        boundary = right_index + right_word.len();
        let _ = left_index;
    }
    boundary
}

trait WordBoundaries {
    fn split_word_bound_indices(&self) -> Vec<(usize, &str)>;
}

impl WordBoundaries for str {
    fn split_word_bound_indices(&self) -> Vec<(usize, &str)> {
        let mut words = Vec::new();
        let mut start = None;
        for (index, character) in self.char_indices() {
            if character.is_whitespace() {
                if let Some(word_start) = start.take() {
                    words.push((word_start, &self[word_start..index]));
                }
            } else if start.is_none() {
                start = Some(index);
            }
        }
        if let Some(word_start) = start {
            words.push((word_start, &self[word_start..]));
        }
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commits_only_matching_word_prefixes() {
        let mut stabilizer = TranscriptStabilizer::default();
        assert_eq!(
            stabilizer.update("hello wor").unwrap().stable_prefix_bytes,
            0
        );
        let update = stabilizer.update("hello world today").unwrap();
        assert_eq!(&update.text[..update.stable_prefix_bytes], "hello");
    }

    #[test]
    fn suppresses_empty_and_duplicate_updates() {
        let mut stabilizer = TranscriptStabilizer::default();
        assert!(stabilizer.update(" ").is_none());
        assert!(stabilizer.update("hello").is_some());
        assert!(stabilizer.update("hello").is_none());
    }
}
