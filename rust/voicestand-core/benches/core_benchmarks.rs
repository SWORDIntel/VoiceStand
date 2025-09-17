use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use voicestand_core::{AppState, VoiceStandConfig, AppEvent};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Benchmark the core AppState operations
fn benchmark_app_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("app_state");

    // Create a test configuration
    let config = VoiceStandConfig::default();
    let app_state = AppState::new(config).unwrap();

    group.bench_function("recording_state_toggle", |b| {
        b.iter(|| {
            app_state.set_recording(black_box(true));
            app_state.set_recording(black_box(false));
        })
    });

    group.bench_function("get_recording_state", |b| {
        b.iter(|| {
            black_box(app_state.is_recording());
        })
    });

    group.bench_function("send_event", |b| {
        b.iter(|| {
            let _ = app_state.send_event(black_box(AppEvent::RecordingStateChanged(true)));
        })
    });

    group.finish();
}

// Benchmark cross-thread communication
fn benchmark_event_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_system");
    group.throughput(Throughput::Elements(1));

    let config = VoiceStandConfig::default();
    let app_state = AppState::new(config).unwrap();

    // Test different event types
    let events = vec![
        AppEvent::RecordingStateChanged(true),
        AppEvent::AudioLevelChanged(0.75),
        AppEvent::TranscriptionReceived("test text".to_string()),
        AppEvent::ErrorOccurred("test error".to_string()),
    ];

    for (i, event) in events.into_iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("send_event", i), &event, |b, event| {
            b.iter(|| {
                let _ = app_state.send_event(black_box(event.clone()));
            });
        });
    }

    group.finish();
}

// Benchmark memory allocation patterns
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Test Vec allocation patterns
    group.bench_function("vec_allocation_small", |b| {
        b.iter(|| {
            let _vec: Vec<f32> = black_box(Vec::with_capacity(1024)); // 1KB
        })
    });

    group.bench_function("vec_allocation_medium", |b| {
        b.iter(|| {
            let _vec: Vec<f32> = black_box(Vec::with_capacity(16384)); // 64KB
        })
    });

    group.bench_function("vec_allocation_large", |b| {
        b.iter(|| {
            let _vec: Vec<f32> = black_box(Vec::with_capacity(65536)); // 256KB
        })
    });

    // Test pre-allocated vs dynamic allocation
    let mut pre_allocated = Vec::with_capacity(16384);
    group.bench_function("vec_reuse_preallocated", |b| {
        b.iter(|| {
            pre_allocated.clear();
            pre_allocated.extend((0..1024).map(|i| i as f32));
            black_box(&pre_allocated);
        })
    });

    group.bench_function("vec_dynamic_allocation", |b| {
        b.iter(|| {
            let vec: Vec<f32> = (0..1024).map(|i| i as f32).collect();
            black_box(vec);
        })
    });

    group.finish();
}

// Benchmark audio sample processing patterns
fn benchmark_audio_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_processing");
    group.throughput(Throughput::Elements(16000)); // 1 second at 16kHz

    // Create test audio data
    let mut samples_16k: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut samples_48k: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.1).sin()).collect();

    // Benchmark basic operations
    group.bench_function("gain_scalar_16k", |b| {
        b.iter(|| {
            for sample in samples_16k.iter_mut() {
                *sample *= black_box(0.8);
            }
        })
    });

    group.bench_function("gain_iterator_16k", |b| {
        b.iter(|| {
            samples_16k.iter_mut().for_each(|sample| *sample *= black_box(0.8));
        })
    });

    // Test different sample rates
    group.throughput(Throughput::Elements(48000));
    group.bench_function("gain_scalar_48k", |b| {
        b.iter(|| {
            for sample in samples_48k.iter_mut() {
                *sample *= black_box(0.8);
            }
        })
    });

    // Benchmark noise gate operation
    group.throughput(Throughput::Elements(16000));
    group.bench_function("noise_gate_16k", |b| {
        let threshold = 0.1;
        let reduction = 0.5;
        b.iter(|| {
            for sample in samples_16k.iter_mut() {
                if sample.abs() < black_box(threshold) {
                    *sample *= black_box(reduction);
                }
            }
        })
    });

    // Benchmark RMS calculation
    group.bench_function("rms_calculation_16k", |b| {
        b.iter(|| {
            let sum_squares: f32 = samples_16k.iter().map(|&x| x * x).sum();
            black_box((sum_squares / samples_16k.len() as f32).sqrt());
        })
    });

    group.finish();
}

// Benchmark lock contention patterns
fn benchmark_lock_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock_contention");

    use parking_lot::Mutex;
    use std::sync::Arc;

    let shared_data = Arc::new(Mutex::new(0u64));

    group.bench_function("single_thread_mutex", |b| {
        let data = Arc::clone(&shared_data);
        b.iter(|| {
            let mut guard = data.lock();
            *guard += black_box(1);
        })
    });

    // Simulate multi-threaded contention
    group.bench_function("multi_thread_contention", |b| {
        let data = Arc::clone(&shared_data);
        b.iter(|| {
            // Simulate brief contention
            let handles: Vec<_> = (0..4).map(|_| {
                let data_clone = Arc::clone(&data);
                std::thread::spawn(move || {
                    let mut guard = data_clone.lock();
                    *guard += 1;
                })
            }).collect();

            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    // Compare with atomic operations
    use std::sync::atomic::{AtomicU64, Ordering};
    let atomic_data = Arc::new(AtomicU64::new(0));

    group.bench_function("atomic_operation", |b| {
        let data = Arc::clone(&atomic_data);
        b.iter(|| {
            data.fetch_add(black_box(1), Ordering::Relaxed);
        })
    });

    group.finish();
}

// Benchmark CPU feature detection
#[cfg(target_arch = "x86_64")]
fn benchmark_cpu_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_features");

    group.bench_function("feature_detection", |b| {
        b.iter(|| {
            black_box(is_x86_feature_detected!("avx2"));
            black_box(is_x86_feature_detected!("fma"));
            black_box(is_x86_feature_detected!("bmi1"));
            black_box(is_x86_feature_detected!("bmi2"));
        })
    });

    // Benchmark SIMD vs scalar operations
    let mut data = vec![1.0f32; 1024];

    group.bench_function("scalar_multiply", |b| {
        b.iter(|| {
            for sample in data.iter_mut() {
                *sample *= black_box(0.8);
            }
        })
    });

    // AVX2 benchmark (if available)
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2_multiply", |b| {
            b.iter(|| {
                unsafe {
                    avx2_multiply(&mut data, black_box(0.8));
                }
            })
        });
    }

    group.finish();
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_multiply(samples: &mut [f32], multiplier: f32) {
    use std::arch::x86_64::*;

    let mult_vec = _mm256_set1_ps(multiplier);

    for chunk in samples.chunks_exact_mut(8) {
        let data = _mm256_loadu_ps(chunk.as_ptr());
        let result = _mm256_mul_ps(data, mult_vec);
        _mm256_storeu_ps(chunk.as_mut_ptr(), result);
    }
}

// Performance regression tests
fn benchmark_performance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_regression");

    // Baseline performance measurements
    let config = VoiceStandConfig::default();
    let app_state = AppState::new(config).unwrap();

    // This should complete within specific time bounds
    group.bench_function("app_initialization", |b| {
        b.iter(|| {
            let config = black_box(VoiceStandConfig::default());
            let _state = black_box(AppState::new(config).unwrap());
        })
    });

    // Memory allocation regression test
    group.bench_function("memory_allocation_regression", |b| {
        b.iter(|| {
            let mut buffers = Vec::new();
            for size in [1024, 4096, 16384, 65536] {
                let buffer: Vec<f32> = black_box(vec![0.0; size]);
                buffers.push(buffer);
            }
            black_box(buffers);
        })
    });

    // Event system regression test
    group.bench_function("event_system_regression", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let _ = app_state.send_event(black_box(AppEvent::AudioLevelChanged(0.5)));
            }
        })
    });

    group.finish();
}

// Custom benchmark configuration
fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(100)
        .with_output_color(true)
}

// Group all benchmarks
criterion_group!(
    name = benches;
    config = criterion_config();
    targets =
        benchmark_app_state,
        benchmark_event_system,
        benchmark_memory_patterns,
        benchmark_audio_processing,
        benchmark_lock_contention,
        benchmark_performance_regression
);

// Add CPU-specific benchmarks for x86_64
#[cfg(target_arch = "x86_64")]
criterion_group!(
    name = x86_benches;
    config = criterion_config();
    targets = benchmark_cpu_features
);

#[cfg(target_arch = "x86_64")]
criterion_main!(benches, x86_benches);

#[cfg(not(target_arch = "x86_64"))]
criterion_main!(benches);