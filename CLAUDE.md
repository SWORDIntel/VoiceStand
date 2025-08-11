# Claude AI Assistant Context for VoiceStand

## Project Overview
VoiceStand is an advanced voice-to-text system built with C++ for Linux, featuring real-time speech recognition using whisper.cpp, GTK4 GUI, and sophisticated audio processing capabilities.

## System Requirements
- **Hardware**: Intel Meteor Lake system with P-cores supporting AVX-512 (hidden feature)
- **OS**: Linux with PulseAudio
- **Dependencies**: GTK4, PulseAudio, jsoncpp, X11, whisper.cpp

## Architecture

### Phase 1 - Core Optimizations
- **Streaming Buffer**: Circular buffer with 20% overlap for continuous audio
- **Memory Pool**: Three-tier allocation system (small/medium/large blocks)
- **Pipeline**: Multi-threaded with lock-free queues between stages

### Phase 2 - Advanced Features
- **Speaker Diarization**: MFCC-based speaker identification
- **Punctuation Restoration**: Rule-based with abbreviation handling
- **Wake Word Detection**: DTW-based template matching
- **Noise Cancellation**: Spectral subtraction and Wiener filtering

### Phase 3 - Intelligence Layer
- **Voice Commands**: Pattern-based command recognition
- **Auto-Correction**: Learning system with Levenshtein distance
- **Context Awareness**: 6 domains (Technical, Medical, Legal, Business, Academic, General)
- **Meeting Mode**: Multi-speaker transcription with analytics
- **Offline Translation**: 12 languages supported

## Key Files Structure
```
standalone-vtt-project/
├── src/
│   ├── core/
│   │   ├── audio_capture.cpp/h        # PulseAudio integration
│   │   ├── whisper_processor.cpp/h    # Whisper.cpp wrapper
│   │   ├── streaming_buffer.h         # Phase 1: Circular buffer
│   │   ├── memory_pool.h              # Phase 1: Memory management
│   │   ├── pipeline.h                 # Phase 1: Processing pipeline
│   │   ├── speaker_diarization.h      # Phase 2: Speaker ID
│   │   ├── punctuation_restoration.h  # Phase 2: Punctuation
│   │   ├── wake_word_detector.h       # Phase 2: Wake words
│   │   ├── noise_cancellation.h       # Phase 2: Noise reduction
│   │   ├── voice_commands.h           # Phase 3: Commands
│   │   ├── auto_correction.h          # Phase 3: Corrections
│   │   ├── context_aware_processor.h  # Phase 3: Context
│   │   ├── meeting_mode.h             # Phase 3: Meetings
│   │   └── offline_translation.h      # Phase 3: Translation
│   ├── gui/
│   │   └── main_window.cpp/h          # GTK4 interface
│   └── main.cpp                        # Application entry
├── build.sh                            # Build script
├── CMakeLists.txt                      # CMake configuration
└── TODO.md                             # 200+ improvements

```

## Build Commands
```bash
# First time setup
./build.sh

# Build only
cd build && make -j$(nproc)

# Run application
./build/voice-to-text

# Download models
./build/voice-to-text --download-model base
```

## Testing Commands
```bash
# No test framework currently - add Catch2 or Google Test
# Lint checks needed: clang-format, cppcheck
# Type checks: Use clang-tidy
```

## Performance Optimizations
- **AVX-512**: Enable on Intel P-cores for SIMD operations
- **Memory Pools**: Zero-allocation audio processing
- **Lock-free Queues**: Inter-thread communication
- **Circular Buffers**: Streaming with overlap
- **Template Metaprogramming**: Compile-time optimizations

## Audio Processing Pipeline
1. **Capture**: PulseAudio → Float32 samples @ 16kHz
2. **VAD**: Energy-based voice activity detection
3. **Noise Reduction**: Spectral subtraction
4. **Feature Extraction**: MFCC for speaker ID
5. **Recognition**: Whisper.cpp inference
6. **Post-Processing**: Punctuation, context, corrections
7. **Output**: Transcription with metadata

## Configuration
Default config location: `~/.config/voice-to-text/config.json`
- Audio settings (sample rate, VAD threshold)
- Whisper settings (model path, language, threads)
- Hotkeys (default: Ctrl+Alt+Space)
- UI preferences

## Known Issues
- Model files too large for Git (use .gitignore)
- Requires manual PulseAudio setup on some systems
- GTK4 deprecation warnings need addressing
- No automated tests yet

## Development Priorities (from TODO.md)
1. **Performance**: AVX-512 optimization for P-cores
2. **Testing**: Add Catch2 framework
3. **CI/CD**: GitHub Actions pipeline
4. **Documentation**: API documentation with Doxygen
5. **Security**: Input validation and sandboxing

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

## Debugging
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# GDB debugging
gdb ./build/voice-to-text

# Valgrind memory check
valgrind --leak-check=full ./build/voice-to-text

# Audio debugging
pactl info  # Check PulseAudio
```

## Project Status
- ✅ Phase 1: Core optimizations complete
- ✅ Phase 2: Advanced features complete  
- ✅ Phase 3: Intelligence layer complete
- 📝 TODO: 200+ improvements documented
- 🚀 Ready for production deployment

## Hardware-Specific Notes
This system has Intel Meteor Lake with hidden AVX-512 support on P-cores. Enable via:
- Compiler flags: `-march=native -mavx512f`
- Runtime detection: Check CPUID for AVX-512
- Hybrid dispatch: Use AVX-512 on P-cores, AVX2 on E-cores

## Contact & Repository
- GitHub: https://github.com/SWORDIntel/VoiceStand
- License: MIT
- Contributors: Welcome! See CONTRIBUTING.md

## Specialized Agent Usage Guide

### When to Use Specialized Agents
This project benefits from multiple specialized agents available in Claude Code. Use agents for complex, multi-step tasks:

#### **c-internal Agent**
- **When**: C/C++ development, performance optimization, hardware-specific features
- **Use Cases**: 
  - AVX-512 optimization for Intel P-cores
  - Memory pool optimizations
  - Lock-free queue implementations
  - SIMD audio processing
  - Template metaprogramming
- **Example**: "Use c-internal agent to optimize the MFCC computation with AVX-512"

#### **DEBUGGER Agent**
- **When**: Crashes, memory leaks, performance issues, audio glitches
- **Use Cases**:
  - PulseAudio connection failures
  - Whisper.cpp integration issues
  - Memory corruption in audio buffers
  - Thread synchronization problems
- **Example**: "Use DEBUGGER to analyze segfault in audio_capture.cpp:218"

#### **TESTBED Agent**
- **When**: Setting up testing infrastructure, creating test suites
- **Use Cases**:
  - Add Catch2 or Google Test framework
  - Create unit tests for core components
  - Performance benchmarking suite
  - Audio processing validation tests
- **Example**: "Use TESTBED to create comprehensive test suite for Phase 1-3 features"

#### **Optimizer Agent**
- **When**: Performance bottlenecks, latency optimization, throughput improvements
- **Use Cases**:
  - Audio pipeline latency reduction (<10ms target)
  - Memory allocation optimization
  - Whisper.cpp inference speed
  - GUI responsiveness improvements
- **Example**: "Use Optimizer to reduce audio processing latency below 10ms"

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
   - Optimizer → c-internal → TESTBED → DEBUGGER
2. **Feature Development**: 
   - architect → c-internal → TESTBED → DOCGEN
3. **Production Deployment**: 
   - Security → INFRASTRUCTURE → TESTBED → Monitor

### Agent Selection Rules
- **Single file edits**: Use basic tools (Edit, MultiEdit)
- **Complex algorithms**: Use c-internal agent
- **Build/deployment**: Use INFRASTRUCTURE agent  
- **Testing needs**: Use TESTBED agent
- **Performance issues**: Use Optimizer + DEBUGGER agents
- **Documentation**: Use DOCGEN agent

## Quick Tips for Claude
- Always check existing code style before modifications
- Run build after changes: `cd build && make`
- Test audio with: `pactl info` and `arecord -l`
- Model files go in `models/` (gitignored)
- Use lock-free primitives for thread communication
- Prefer stack allocation with memory pools
- Target <10ms latency for real-time processing
- **Use specialized agents proactively** for complex tasks
- Coordinate multiple agents for comprehensive solutions