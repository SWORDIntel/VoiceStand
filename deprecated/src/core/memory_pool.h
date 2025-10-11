#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <mutex>
#include <atomic>

namespace vtt {

// Memory pool for zero-allocation audio processing after initialization
template<typename T>
class MemoryPool {
public:
    struct Block {
        std::vector<T> data;
        std::atomic<bool> in_use{false};
        
        explicit Block(size_t size) : data(size) {}
        
        void reset() {
            in_use = false;
        }
    };
    
    using BlockPtr = std::shared_ptr<Block>;
    
    MemoryPool(size_t block_size, size_t num_blocks)
        : block_size_(block_size) {
        
        // Pre-allocate all blocks
        for (size_t i = 0; i < num_blocks; ++i) {
            auto block = std::make_shared<Block>(block_size);
            free_blocks_.push(block);
            all_blocks_.push_back(block);
        }
    }
    
    // Get a free block from the pool
    BlockPtr acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (free_blocks_.empty()) {
            // All blocks in use - allocate new one (only happens under high load)
            auto block = std::make_shared<Block>(block_size_);
            all_blocks_.push_back(block);
            stats_.overflow_count++;
            return block;
        }
        
        auto block = free_blocks_.front();
        free_blocks_.pop();
        block->in_use = true;
        stats_.acquisitions++;
        return block;
    }
    
    // Return a block to the pool
    void release(BlockPtr block) {
        if (!block || !block->in_use) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        block->reset();
        free_blocks_.push(block);
        stats_.releases++;
    }
    
    // Get pool statistics
    struct Stats {
        std::atomic<size_t> acquisitions{0};
        std::atomic<size_t> releases{0};
        std::atomic<size_t> overflow_count{0};
        
        // Delete copy constructor and assignment to prevent atomic copy issues
        Stats() = default;
        Stats(const Stats&) = delete;
        Stats& operator=(const Stats&) = delete;
    };
    
    const Stats& get_stats() const { return stats_; }
    
    size_t available_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_.size();
    }
    
    size_t total_blocks() const {
        return all_blocks_.size();
    }
    
private:
    const size_t block_size_;
    std::queue<BlockPtr> free_blocks_;
    std::vector<BlockPtr> all_blocks_;
    mutable std::mutex mutex_;
    Stats stats_;
};

// Specialized audio buffer pool
class AudioBufferPool {
public:
    static constexpr size_t SMALL_BUFFER = 1024;    // 64ms at 16kHz
    static constexpr size_t MEDIUM_BUFFER = 4096;   // 256ms at 16kHz
    static constexpr size_t LARGE_BUFFER = 16384;   // 1s at 16kHz
    
    AudioBufferPool() 
        : small_pool_(SMALL_BUFFER, 32)
        , medium_pool_(MEDIUM_BUFFER, 16)
        , large_pool_(LARGE_BUFFER, 8) {
    }
    
    MemoryPool<float>::BlockPtr get_buffer(size_t size) {
        if (size <= SMALL_BUFFER) {
            return small_pool_.acquire();
        } else if (size <= MEDIUM_BUFFER) {
            return medium_pool_.acquire();
        } else {
            return large_pool_.acquire();
        }
    }
    
    void return_buffer(MemoryPool<float>::BlockPtr buffer) {
        if (!buffer) return;
        
        size_t size = buffer->data.size();
        if (size <= SMALL_BUFFER) {
            small_pool_.release(buffer);
        } else if (size <= MEDIUM_BUFFER) {
            medium_pool_.release(buffer);
        } else {
            large_pool_.release(buffer);
        }
    }
    
    void print_stats() const {
        auto print_pool_stats = [](const char* name, const auto& pool) {
            const auto& stats = pool.get_stats();
            printf("%s Pool - Acquisitions: %zu, Releases: %zu, Overflows: %zu, Available: %zu/%zu\n",
                   name,
                   stats.acquisitions.load(),
                   stats.releases.load(),
                   stats.overflow_count.load(),
                   pool.available_blocks(),
                   pool.total_blocks());
        };
        
        print_pool_stats("Small", small_pool_);
        print_pool_stats("Medium", medium_pool_);
        print_pool_stats("Large", large_pool_);
    }
    
private:
    MemoryPool<float> small_pool_;
    MemoryPool<float> medium_pool_;
    MemoryPool<float> large_pool_;
};

}  // namespace vtt