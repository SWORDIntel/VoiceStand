use std::collections::VecDeque;
use parking_lot::Mutex;
use voicestand_core::{Result, VoiceStandError};

/// Thread-safe circular buffer for audio samples
pub struct CircularBuffer<T> {
    inner: Mutex<CircularBufferInner<T>>,
}

struct CircularBufferInner<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    overflow_count: usize,
}

impl<T: Clone> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(CircularBufferInner {
                buffer: VecDeque::with_capacity(capacity),
                capacity,
                overflow_count: 0,
            }),
        }
    }

    /// Write data to the buffer
    pub fn write(&self, data: &[T]) -> Result<usize> {
        let mut inner = self.inner.lock();
        let mut written = 0;

        for item in data {
            if inner.buffer.len() >= inner.capacity {
                // Remove oldest item to make space
                inner.buffer.pop_front();
                inner.overflow_count += 1;
            }
            inner.buffer.push_back(item.clone());
            written += 1;
        }

        Ok(written)
    }

    /// Read data from the buffer
    pub fn read(&self, data: &mut [T]) -> Result<usize> {
        let mut inner = self.inner.lock();
        let to_read = std::cmp::min(data.len(), inner.buffer.len());

        for i in 0..to_read {
            if let Some(item) = inner.buffer.pop_front() {
                data[i] = item;
            }
        }

        Ok(to_read)
    }

    /// Read all available data
    pub fn read_all(&self) -> Vec<T> {
        let mut inner = self.inner.lock();
        inner.buffer.drain(..).collect()
    }

    /// Get the number of available items
    pub fn available(&self) -> usize {
        self.inner.lock().buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.inner.lock().buffer.is_empty()
    }

    /// Clear the buffer
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.buffer.clear();
    }

    /// Get overflow count (number of samples dropped due to buffer full)
    pub fn overflow_count(&self) -> usize {
        self.inner.lock().overflow_count
    }

    /// Reset overflow count
    pub fn reset_overflow_count(&self) {
        self.inner.lock().overflow_count = 0;
    }
}

/// Streaming buffer with overlap for continuous processing
pub struct StreamingBuffer {
    buffer: CircularBuffer<f32>,
    chunk_size: usize,
    overlap_size: usize,
    last_chunk: Vec<f32>,
}

impl StreamingBuffer {
    pub fn new(chunk_size: usize, overlap_percent: f32) -> Self {
        let overlap_size = (chunk_size as f32 * overlap_percent) as usize;
        let buffer_capacity = chunk_size * 4; // Buffer 4 chunks

        Self {
            buffer: CircularBuffer::new(buffer_capacity),
            chunk_size,
            overlap_size,
            last_chunk: Vec::new(),
        }
    }

    /// Add audio samples to the buffer
    pub fn push(&self, samples: &[f32]) -> Result<()> {
        self.buffer.write(samples)?;
        Ok(())
    }

    /// Get the next chunk with overlap
    pub fn pop_chunk(&mut self) -> Option<Vec<f32>> {
        let available = self.buffer.available();
        if available < self.chunk_size {
            return None;
        }

        let mut chunk = vec![0.0; self.chunk_size];
        match self.buffer.read(&mut chunk) {
            Ok(bytes_read) if bytes_read == self.chunk_size => {
            // Add overlap from previous chunk
            if !self.last_chunk.is_empty() && self.overlap_size > 0 {
                let overlap_start = self.last_chunk.len().saturating_sub(self.overlap_size);
                let overlap_data = &self.last_chunk[overlap_start..];

                // Blend overlap
                let blend_size = std::cmp::min(overlap_data.len(), chunk.len());
                for i in 0..blend_size {
                    chunk[i] = (chunk[i] + overlap_data[i]) * 0.5;
                }
            }

                self.last_chunk = chunk.clone();
                Some(chunk)
            }
            Ok(_) => None, // Not enough bytes read
            Err(_) => None, // Read error
        }
    }

    /// Clear the buffer and reset state
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_chunk.clear();
    }

    /// Get buffer statistics
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            available_samples: self.buffer.available(),
            overflow_count: self.buffer.overflow_count(),
            chunk_size: self.chunk_size,
            overlap_size: self.overlap_size,
        }
    }
}

/// Buffer statistics for monitoring
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub available_samples: usize,
    pub overflow_count: usize,
    pub chunk_size: usize,
    pub overlap_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_buffer() {
        let buffer = CircularBuffer::new(5);

        // Write some data
        assert_eq!(buffer.write(&[1, 2, 3]).expect("Write should succeed"), 3);
        assert_eq!(buffer.available(), 3);

        // Read some data
        let mut read_data = [0; 2];
        assert_eq!(buffer.read(&mut read_data).expect("Read should succeed"), 2);
        assert_eq!(read_data, [1, 2]);
        assert_eq!(buffer.available(), 1);

        // Test overflow
        assert_eq!(buffer.write(&[4, 5, 6, 7, 8, 9]).expect("Write should succeed"), 6);
        assert_eq!(buffer.overflow_count(), 2); // 2 items were dropped
    }

    #[test]
    fn test_streaming_buffer() {
        let mut buffer = StreamingBuffer::new(4, 0.25); // 25% overlap = 1 sample

        // Add samples
        buffer.push(&[1.0, 2.0, 3.0, 4.0]).expect("Push should succeed");
        buffer.push(&[5.0, 6.0, 7.0, 8.0]).expect("Push should succeed");

        // Get first chunk
        let chunk1 = buffer.pop_chunk().expect("Should get first chunk");
        assert_eq!(chunk1.len(), 4);

        // Get second chunk (should have overlap)
        let chunk2 = buffer.pop_chunk();
        assert!(chunk2.is_some());
    }
}