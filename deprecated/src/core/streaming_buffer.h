#pragma once

#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>

namespace vtt {

// Circular buffer optimized for audio streaming with overlap support
template<typename T>
class StreamingBuffer {
public:
    explicit StreamingBuffer(size_t capacity, float overlap_ratio = 0.2f)
        : capacity_(capacity)
        , overlap_size_(static_cast<size_t>(capacity * overlap_ratio))
        , buffer_(capacity)
        , write_pos_(0)
        , read_pos_(0)
        , size_(0) {
    }
    
    // Write samples to buffer
    bool write(const T* data, size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size_ + count > capacity_) {
            // Buffer overflow - drop oldest data
            size_t to_drop = (size_ + count) - capacity_;
            read_pos_ = (read_pos_ + to_drop) % capacity_;
            size_ -= to_drop;
        }
        
        for (size_t i = 0; i < count; ++i) {
            buffer_[write_pos_] = data[i];
            write_pos_ = (write_pos_ + 1) % capacity_;
        }
        
        size_ += count;
        return true;
    }
    
    // Read chunk with overlap from previous chunk
    std::vector<T> read_chunk_with_overlap(size_t chunk_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size_ < chunk_size) {
            return {};  // Not enough data
        }
        
        std::vector<T> chunk;
        chunk.reserve(chunk_size + overlap_size_);
        
        // Include overlap from previous chunk if available
        size_t overlap_start = (read_pos_ >= overlap_size_) 
            ? read_pos_ - overlap_size_ 
            : capacity_ - (overlap_size_ - read_pos_);
            
        if (last_chunk_read_ && overlap_size_ > 0) {
            // Add overlap from previous chunk
            for (size_t i = 0; i < overlap_size_; ++i) {
                size_t pos = (overlap_start + i) % capacity_;
                chunk.push_back(buffer_[pos]);
            }
        }
        
        // Read main chunk
        for (size_t i = 0; i < chunk_size; ++i) {
            chunk.push_back(buffer_[read_pos_]);
            read_pos_ = (read_pos_ + 1) % capacity_;
        }
        
        size_ -= chunk_size;
        last_chunk_read_ = true;
        
        return chunk;
    }
    
    size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        write_pos_ = read_pos_ = size_ = 0;
        last_chunk_read_ = false;
    }
    
private:
    const size_t capacity_;
    const size_t overlap_size_;
    std::vector<T> buffer_;
    size_t write_pos_;
    size_t read_pos_;
    std::atomic<size_t> size_;
    bool last_chunk_read_ = false;
    mutable std::mutex mutex_;
};

}  // namespace vtt