#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>
#include "memory_pool.h"
#include "streaming_buffer.h"

namespace vtt {

// Lock-free queue for passing data between pipeline stages
template<typename T>
class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity = 1024) 
        : capacity_(capacity)
        , buffer_(capacity)
        , head_(0)
        , tail_(0) {
    }
    
    bool push(T item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % capacity_;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;  // Queue full
        }
        
        buffer_[current_tail] = std::move(item);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Queue empty
        }
        
        item = std::move(buffer_[current_head]);
        head_.store((current_head + 1) % capacity_, std::memory_order_release);
        return true;
    }
    
    bool empty() const {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }
    
private:
    const size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};

// Pipeline stage for audio processing
class PipelineStage {
public:
    using ProcessFunc = std::function<void(MemoryPool<float>::BlockPtr)>;
    
    PipelineStage(const std::string& name, ProcessFunc func)
        : name_(name)
        , process_func_(func)
        , running_(false) {
    }
    
    void start() {
        running_ = true;
        worker_ = std::thread(&PipelineStage::worker_loop, this);
    }
    
    void stop() {
        running_ = false;
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }
    
    bool enqueue(MemoryPool<float>::BlockPtr data) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (input_queue_.size() >= max_queue_size_) {
                stats_.dropped_frames++;
                return false;
            }
            input_queue_.push(data);
        }
        cv_.notify_one();
        return true;
    }
    
    struct Stats {
        std::atomic<size_t> processed_frames{0};
        std::atomic<size_t> dropped_frames{0};
        std::atomic<size_t> total_latency_us{0};
        std::atomic<size_t> max_latency_us{0};
        
        // Delete copy constructor and assignment to prevent atomic copy issues
        Stats() = default;
        Stats(const Stats&) = delete;
        Stats& operator=(const Stats&) = delete;
    };
    
    const Stats& get_stats() const { return stats_; }
    const std::string& name() const { return name_; }
    
private:
    void worker_loop() {
        while (running_) {
            MemoryPool<float>::BlockPtr data;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { 
                    return !input_queue_.empty() || !running_; 
                });
                
                if (!running_) break;
                if (input_queue_.empty()) continue;
                
                data = input_queue_.front();
                input_queue_.pop();
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            process_func_(data);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            stats_.processed_frames++;
            stats_.total_latency_us += latency;
            
            size_t max_lat = stats_.max_latency_us.load();
            while (latency > max_lat && 
                   !stats_.max_latency_us.compare_exchange_weak(max_lat, latency));
        }
    }
    
    std::string name_;
    ProcessFunc process_func_;
    std::thread worker_;
    std::queue<MemoryPool<float>::BlockPtr> input_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_;
    Stats stats_;
    static constexpr size_t max_queue_size_ = 32;
};

// Multi-threaded audio processing pipeline
class AudioPipeline {
public:
    AudioPipeline() 
        : buffer_pool_(std::make_shared<AudioBufferPool>())
        , streaming_buffer_(16384 * 4, 0.2f) {  // 4 seconds buffer with 20% overlap
    }
    
    void add_stage(const std::string& name, PipelineStage::ProcessFunc func) {
        stages_.emplace_back(std::make_unique<PipelineStage>(name, func));
    }
    
    void start() {
        for (auto& stage : stages_) {
            stage->start();
        }
        running_ = true;
        
        // Start monitoring thread
        monitor_thread_ = std::thread(&AudioPipeline::monitor_loop, this);
    }
    
    void stop() {
        running_ = false;
        
        for (auto& stage : stages_) {
            stage->stop();
        }
        
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
    
    void process_audio(const float* samples, size_t num_samples) {
        // Write to streaming buffer
        streaming_buffer_.write(samples, num_samples);
        
        // Check if we have enough data for a chunk
        const size_t chunk_size = 16384;  // 1 second at 16kHz
        if (streaming_buffer_.available() >= chunk_size) {
            auto chunk = streaming_buffer_.read_chunk_with_overlap(chunk_size);
            
            // Get buffer from pool
            auto buffer = buffer_pool_->get_buffer(chunk.size());
            std::copy(chunk.begin(), chunk.end(), buffer->data.begin());
            
            // Send to first stage
            if (!stages_.empty()) {
                stages_[0]->enqueue(buffer);
            }
        }
    }
    
    std::shared_ptr<AudioBufferPool> get_buffer_pool() { return buffer_pool_; }
    
    void print_stats() const {
        printf("\n=== Pipeline Statistics ===\n");
        for (const auto& stage : stages_) {
            const auto& stats = stage->get_stats();
            size_t avg_latency = stats.processed_frames > 0 
                ? stats.total_latency_us / stats.processed_frames 
                : 0;
            
            printf("Stage '%s': Processed=%zu, Dropped=%zu, AvgLatency=%zuus, MaxLatency=%zuus\n",
                   stage->name().c_str(),
                   stats.processed_frames.load(),
                   stats.dropped_frames.load(),
                   avg_latency,
                   stats.max_latency_us.load());
        }
        
        printf("\n=== Buffer Pool Statistics ===\n");
        buffer_pool_->print_stats();
    }
    
private:
    void monitor_loop() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            if (running_) {
                print_stats();
            }
        }
    }
    
    std::vector<std::unique_ptr<PipelineStage>> stages_;
    std::shared_ptr<AudioBufferPool> buffer_pool_;
    StreamingBuffer<float> streaming_buffer_;
    std::atomic<bool> running_{false};
    std::thread monitor_thread_;
};

}  // namespace vtt