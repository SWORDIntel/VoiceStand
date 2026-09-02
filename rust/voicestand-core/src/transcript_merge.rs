/// Merge a cached transcript with a tail decode by matching the longest word
/// suffix of the cached text to a prefix of the overlapping tail text.
pub fn merge_word_overlap(cached: &str, tail: &str) -> Option<String> {
    let cached_words: Vec<&str> = cached.split_whitespace().collect();
    let tail_words: Vec<&str> = tail.split_whitespace().collect();
    if cached_words.is_empty() || tail_words.is_empty() {
        return None;
    }

    let maximum = cached_words.len().min(tail_words.len());
    let overlap = (1..=maximum).rev().find(|&count| {
        cached_words[cached_words.len() - count..]
            .iter()
            .zip(&tail_words[..count])
            .all(|(left, right)| normalize_word(left) == normalize_word(right))
    })?;

    let mut merged = cached.trim().to_string();
    if overlap < tail_words.len() {
        if !merged.is_empty() {
            merged.push(' ');
        }
        merged.push_str(&tail_words[overlap..].join(" "));
    }
    Some(merged)
}

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_the_longest_case_insensitive_word_overlap() {
        assert_eq!(
            merge_word_overlap("Hello brave world", "Brave world, today"),
            Some("Hello brave world today".to_string())
        );
    }

    #[test]
    fn refuses_to_guess_without_an_overlap() {
        assert_eq!(merge_word_overlap("hello world", "different tail"), None);
    }
}
