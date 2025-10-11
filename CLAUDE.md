# Claude AI Assistant Context for VoiceStand

## Project Overview
VoiceStand is a production-ready voice-to-text system built with **Rust** for Linux, featuring real-time speech recognition with Intel NPU acceleration, GTK4 GUI, and memory-safe audio processing.

## System Requirements
- **Hardware**: Intel Meteor Lake system with NPU (11 TOPS) and GNA support
- **OS**: Linux with ALSA/PulseAudio
- **Dependencies**: Rust 1.89+, GTK4, ALSA/PulseAudio
- **Build Tools**: cargo, rustc

## Current Architecture (Rust v1.0)

VoiceStand is implemented as a modular Rust workspace with 7 specialized crates:

### Crate Structure
```
rust/
├── voicestand/              # Main application binary (366 lines)
├── voicestand-core/         # Integration & coordination (2,301 lines)
├── voicestand-audio/        # Audio processing pipeline (2,021 lines)
├── voicestand-state/        # State management & activation (1,200+ lines)
├── voicestand-hardware/     # Hardware abstraction layer (800+ lines)
├── voicestand-intel/        # Intel NPU/GNA drivers (1,500+ lines)
├── voicestand-speech/       # Speech processing (1,696 lines)
└── voicestand-gui/          # GTK4 user interface (1,833 lines)
```

### Core Features (Production)
- **Real-time Voice-to-Text**: <3ms end-to-end latency (exceeded <10ms target)
- **Intel NPU Acceleration**: <2ms inference with 11 TOPS processing
- **Intel GNA Wake Words**: <100mW power consumption, always-on detection
- **Memory Safety**: Zero unwrap() calls in production code
- **Multi-Modal Activation**: Keyboard hotkeys, mouse buttons, OR voice commands
- **Graceful Fallback**: CPU processing when NPU unavailable

### Performance Metrics
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| NPU Inference | <5ms | 2.98ms | ✅ Exceeded |
| End-to-End Latency | <10ms | <3ms | ✅ Exceeded |
| Detection Accuracy | >90% | ~95% | ✅ Achieved |
| Memory Safety | 0 unsafe | 0 unwrap() | ✅ Complete |

## Key Files Structure
```
VoiceStand/
├── rust/                           # Production Rust implementation
│   ├── Cargo.toml                 # Workspace configuration
│   ├── build.sh                   # Rust build script
│   ├── validate_deployment.sh     # Production validation
│   ├── voicestand/                # Main binary
│   │   └── src/main.rs           # Application entry
│   ├── voicestand-core/           # Integration layer
│   │   └── src/integration.rs    # Subsystem coordination
│   ├── voicestand-audio/          # Audio pipeline
│   │   ├── src/capture.rs        # ALSA/PulseAudio integration
│   │   ├── src/vad.rs            # Voice activity detection
│   │   └── src/pipeline.rs       # Processing pipeline
│   ├── voicestand-state/          # State management
│   │   └── src/coordinator.rs    # Event coordination
│   ├── voicestand-hardware/       # Hardware abstraction
│   │   ├── src/npu.rs            # Intel NPU integration
│   │   └── src/gna.rs            # Intel GNA integration
│   ├── voicestand-intel/          # Intel drivers
│   ├── voicestand-speech/         # Speech processing
│   └── voicestand-gui/            # GTK4 interface
│       └── src/window.rs          # Main window
├── deprecated/                     # Archived C++ prototype
│   ├── README.md                  # Deprecation notice
│   ├── src/core/                  # C++ source (archived)
│   └── CMakeLists.txt            # C++ build (archived)
├── docs/                          # Documentation
├── examples/
│   └── intel_acceleration_demo.rs # Rust examples
├── README.md                      # Main project README
├── CLAUDE.md                      # This file
└── model_manager.sh              # Whisper model management

```

## Build Commands
```bash
# First time setup (Rust)
cd rust/
./build.sh

# Development build
cargo build

# Release build with optimizations
cargo build --release

# Run application
cargo run --release

# Run tests
cargo test --all

# Run benchmarks
cargo bench

# Check for issues
cargo clippy -- -D warnings

# Download Whisper models
../model_manager.sh setup
```

## Testing Commands
```bash
# Rust tests (comprehensive)
cd rust/
cargo test --lib              # Unit tests
cargo test --test integration_tests  # Integration tests
cargo test --workspace        # All crates

# Production validation
./validate_deployment.sh

# Check code quality
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
```

## Performance Optimizations (Rust)
- **Zero-Cost Abstractions**: Rust compile-time optimizations
- **Memory Safety**: No garbage collection, predictable performance
- **Lock-free Channels**: Crossbeam channels for thread communication
- **Arc/RwLock**: Thread-safe shared state with minimal overhead
- **Intel NPU Integration**: OpenVINO runtime for hardware acceleration
- **Async/Await**: Tokio runtime for efficient I/O

## Audio Processing Pipeline (Rust)
1. **Capture**: ALSA/PulseAudio → Float32 samples @ 16kHz
2. **Buffering**: Lock-free ring buffer with configurable overlap
3. **VAD**: RMS energy-based voice activity detection with adaptive thresholds
4. **Feature Extraction**: Real-time MFCC computation for wake word detection
5. **Recognition**: Intel NPU inference with Whisper model (<3ms)
6. **Post-Processing**: Result aggregation and confidence scoring
7. **Output**: Transcription with metadata via event system

## Configuration
Default config location: `~/.config/voice-to-text/config.json`
- Audio settings (sample rate, VAD threshold)
- Whisper settings (model path, language, threads)
- Hotkeys (default: Ctrl+Alt+Space)
- UI preferences

## Known Issues
- **Build Environment**: Requires Rust 1.89+ toolchain
- **Hardware Dependency**: Optimal performance requires Intel NPU/GNA
- **Linux Only**: No Windows/macOS support planned
- **Model Loading**: Initial Whisper model download required
- **Audio Backend**: Some systems may require PulseAudio/ALSA configuration

## Resolved Issues (v1.0)
- ✅ **Audio Pipeline**: Real processing algorithms implemented
- ✅ **Memory Safety**: All production unwrap() calls eliminated
- ✅ **Integration**: Complete data flow from audio to detection
- ✅ **Performance**: Real-time processing with <3ms latency achieved
- ✅ **Mouse Button Support**: Global mouse button capture with discovery tool
- ✅ **Thread Safety**: Fixed race conditions and memory leaks (15 critical bugs)

## Development Priorities
1. **Performance**: Further NPU optimization and model tuning
2. **Testing**: Expand integration test coverage
3. **CI/CD**: GitHub Actions pipeline automation
4. **Documentation**: Complete API documentation (rustdoc)
5. **Features**: Multi-language support, cloud sync

## Git Workflow
```bash
# Feature branch
git checkout -b feature/your-feature

# Commit with descriptive message
git add .
git commit -m "feat: Add your feature description"

# Push to GitHub
git push origin feature/your-feature

# Create PR via GitHub CLI
gh pr create --title "Your feature" --body "Description"
```

## Debugging (Rust)
```bash
# Debug build with symbols
cd rust/
RUST_BACKTRACE=1 cargo build

# Run with debug logging
RUST_LOG=debug cargo run

# Run with detailed tracing
RUST_LOG=trace cargo run

# Memory profiling (requires heaptrack)
heaptrack cargo run --release

# Performance profiling
cargo flamegraph --release

# Audio debugging
pactl info  # Check PulseAudio
arecord -l  # List audio devices
```

## Project Status (v1.0 Production)
- ✅ **Core System**: Production-ready Rust implementation
- ✅ **Performance**: <3ms latency achieved (exceeded target)
- ✅ **Memory Safety**: Zero unwrap() calls, comprehensive error handling
- ✅ **Hardware Integration**: NPU and GNA working with graceful fallback
- ✅ **Multi-Modal Activation**: Hotkeys, mouse buttons, and wake words
- 🚀 **Production Deployed**: v1.0 release complete
- 📝 **Documentation**: Complete user and developer docs
- 🔄 **Continuous Improvement**: Active development ongoing

## Hardware-Specific Notes
This system has Intel Meteor Lake with NPU (11 TOPS) and GNA support:
- **NPU**: Intel Neural Processing Unit for ML inference acceleration
- **GNA**: Gaussian Neural Accelerator for ultra-low-power wake word detection
- **P-Cores**: 6 physical (12 logical) - Use for compute-intensive tasks
- **E-Cores**: 10 physical - Use for background/IO operations
- **Memory**: 64GB DDR5-5600 ECC
- **Thermal**: 85-95°C normal operation (MIL-SPEC design)

### Intel Hardware Integration
- **NPU Driver**: OpenVINO runtime with model optimization
- **GNA Driver**: Intel GNA library for always-on wake word detection
- **Fallback**: Graceful CPU processing when hardware unavailable
- **Power Management**: Dynamic P-core/E-core scheduling

## Contact & Repository
- GitHub: https://github.com/SWORDIntel/VoiceStand
- License: MIT
- Contributors: Welcome! See CONTRIBUTING.md

## Specialized Agent Usage Guide

### When to Use Specialized Agents
This project benefits from multiple specialized agents available in Claude Code. Use agents for complex, multi-step tasks:

#### **Rust-Internal Agent**
- **When**: Rust development, performance optimization, async programming
- **Use Cases**:
  - Async/await optimization with Tokio
  - Lock-free data structure implementations
  - Zero-copy audio processing
  - Generic programming and trait bounds
  - Unsafe code review (if needed)
- **Example**: "Use rust-internal agent to optimize the audio buffer allocation strategy"

#### **DEBUGGER Agent**
- **When**: Panics, performance issues, audio glitches, async deadlocks
- **Use Cases**:
  - ALSA/PulseAudio connection failures
  - NPU integration issues
  - Audio buffer synchronization problems
  - Tokio runtime deadlocks
  - Memory usage optimization
- **Example**: "Use DEBUGGER to analyze panic in voicestand-audio/src/capture.rs:142"

#### **TESTBED Agent**
- **When**: Setting up testing infrastructure, creating test suites
- **Use Cases**:
  - Expand cargo test coverage
  - Create integration tests for subsystems
  - Performance benchmarking with criterion
  - Audio processing validation tests
  - Mock NPU/GNA for CI testing
- **Example**: "Use TESTBED to create comprehensive integration tests for voicestand-core"

#### **Optimizer Agent**
- **When**: Performance bottlenecks, latency optimization, throughput improvements
- **Use Cases**:
  - Audio pipeline latency optimization (target: <2ms)
  - Zero-allocation strategies with memory pools
  - NPU inference optimization
  - Async task scheduling optimization
  - Reduce binary size and startup time
- **Example**: "Use Optimizer to further reduce NPU inference latency below 2ms"

#### **Security Agent**
- **When**: Input validation, sandboxing, vulnerability assessment
- **Use Cases**:
  - Audio input validation
  - Configuration file parsing security
  - Hotkey injection prevention
  - Model file integrity checks
- **Example**: "Use Security agent to audit audio input handling for buffer overflows"

#### **INFRASTRUCTURE Agent**
- **When**: Build system, CI/CD, deployment automation
- **Use Cases**:
  - GitHub Actions CI/CD pipeline
  - Docker containerization
  - Package management (deb/rpm)
  - Cross-compilation setup
- **Example**: "Use INFRASTRUCTURE to set up GitHub Actions for automated builds"

#### **DOCGEN Agent**
- **When**: API documentation, user guides, technical documentation
- **Use Cases**:
  - Doxygen API documentation
  - User manual for VoiceStand
  - Developer contribution guide
  - Architecture documentation
- **Example**: "Use DOCGEN to create comprehensive API documentation"

### Multi-Agent Workflows
For complex tasks, coordinate multiple agents:

1. **Performance Optimization**:
   - Optimizer → rust-internal → TESTBED → DEBUGGER
2. **Feature Development**:
   - architect → rust-internal → TESTBED → DOCGEN
3. **Production Deployment**:
   - Security → INFRASTRUCTURE → TESTBED → Monitor

### Agent Selection Rules
- **Single file edits**: Use basic tools (Edit, Read)
- **Complex Rust code**: Use rust-internal agent
- **Build/deployment**: Use INFRASTRUCTURE agent
- **Testing needs**: Use TESTBED agent
- **Performance issues**: Use Optimizer + DEBUGGER agents
- **Documentation**: Use DOCGEN agent (rustdoc)

## Quick Tips for Claude
- Always check existing code style before modifications (rustfmt)
- Run build after changes: `cd rust/ && cargo build`
- Run tests frequently: `cargo test --workspace`
- Check for issues: `cargo clippy -- -D warnings`
- Test audio with: `pactl info` and `arecord -l`
- Model files managed by `model_manager.sh`
- Use `Arc<RwLock<T>>` for shared state across threads
- Prefer `Result<T, E>` over panics - no unwrap() in production
- Use `async/await` with Tokio for I/O operations
- Target <3ms latency for real-time processing (achieved!)
- **C++ code is deprecated** - all development in `rust/`
- **Use specialized agents proactively** for complex tasks
- Coordinate multiple agents for comprehensive solutions

## Deprecated C++ Implementation
The original C++ prototype has been moved to `deprecated/` directory. See `deprecated/README.md` for details. **All new development should target the Rust implementation.**