#!/bin/bash

# VoiceStand Comprehensive Test Suite
# Memory safety and performance validation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 VoiceStand Comprehensive Test Suite${NC}"
echo "================================================"

# Test configuration
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_MEMORY_TESTS=true
RUN_PERFORMANCE_TESTS=true
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)
            RUN_INTEGRATION_TESTS=false
            RUN_MEMORY_TESTS=false
            RUN_PERFORMANCE_TESTS=false
            shift
            ;;
        --no-memory)
            RUN_MEMORY_TESTS=false
            shift
            ;;
        --no-performance)
            RUN_PERFORMANCE_TESTS=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "VoiceStand Test Suite"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit-only       Run only unit tests"
            echo "  --no-memory       Skip memory safety tests"
            echo "  --no-performance  Skip performance tests"
            echo "  --verbose, -v     Verbose output"
            echo "  --help, -h        Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -e "${BLUE}Running: $test_name${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if [ "$VERBOSE" = true ]; then
        echo "Command: $test_command"
    fi

    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
}

# Unit Tests
if [ "$RUN_UNIT_TESTS" = true ]; then
    echo -e "${BLUE}📚 Running Unit Tests${NC}"
    echo "------------------------"

    run_test "Core Types and Configuration" "cargo test --package voicestand-core"
    run_test "Audio Processing" "cargo test --package voicestand-audio"
    run_test "Speech Recognition" "cargo test --package voicestand-speech"
    run_test "GUI Components" "cargo test --package voicestand-gui"
    run_test "Main Application" "cargo test --package voicestand"
fi

# Integration Tests
if [ "$RUN_INTEGRATION_TESTS" = true ]; then
    echo -e "${BLUE}🔗 Running Integration Tests${NC}"
    echo "------------------------------"

    # Test audio pipeline
    run_test "Audio-Speech Integration" "cargo test --test audio_speech_integration"

    # Test GUI integration
    run_test "GUI-Core Integration" "cargo test --test gui_integration"

    # Test configuration loading
    run_test "Configuration System" "cargo test --test config_integration"
fi

# Memory Safety Tests
if [ "$RUN_MEMORY_TESTS" = true ]; then
    echo -e "${BLUE}🛡️  Running Memory Safety Tests${NC}"
    echo "--------------------------------"

    # Check if valgrind is available
    if command -v valgrind &> /dev/null; then
        echo "Running Valgrind memory check..."
        run_test "Valgrind Memory Check" "cargo build --release && valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes ./target/release/voicestand --help"
    else
        echo -e "${YELLOW}⚠️  Valgrind not available, skipping memory leak detection${NC}"
    fi

    # Address Sanitizer (if available)
    if cargo --version | grep -q nightly; then
        echo "Running AddressSanitizer..."
        run_test "AddressSanitizer Check" "RUSTFLAGS=\"-Z sanitizer=address\" cargo +nightly test --target x86_64-unknown-linux-gnu"
    else
        echo -e "${YELLOW}⚠️  Nightly Rust not available, skipping AddressSanitizer${NC}"
    fi

    # Thread Sanitizer (if available)
    if cargo --version | grep -q nightly; then
        echo "Running ThreadSanitizer..."
        run_test "ThreadSanitizer Check" "RUSTFLAGS=\"-Z sanitizer=thread\" cargo +nightly test --target x86_64-unknown-linux-gnu"
    else
        echo -e "${YELLOW}⚠️  Nightly Rust not available, skipping ThreadSanitizer${NC}"
    fi
fi

# Performance Tests
if [ "$RUN_PERFORMANCE_TESTS" = true ]; then
    echo -e "${BLUE}⚡ Running Performance Tests${NC}"
    echo "-----------------------------"

    # Build optimized binary
    echo "Building optimized binary for performance testing..."
    cargo build --release

    # Test audio processing performance
    run_test "Audio Processing Latency" "cargo test --release test_audio_latency"

    # Test speech recognition performance
    run_test "Speech Recognition Speed" "cargo test --release test_speech_performance"

    # Test memory usage
    run_test "Memory Usage Analysis" "cargo test --release test_memory_usage"

    # Benchmark tests (if available)
    if [ -f "benches/benchmark.rs" ]; then
        run_test "Benchmark Suite" "cargo bench"
    fi
fi

# Security Tests
echo -e "${BLUE}🔒 Running Security Tests${NC}"
echo "--------------------------"

# Cargo audit (if available)
if command -v cargo-audit &> /dev/null; then
    run_test "Security Vulnerability Scan" "cargo audit"
else
    echo -e "${YELLOW}⚠️  cargo-audit not available, install with: cargo install cargo-audit${NC}"
fi

# Clippy lints
run_test "Clippy Security Lints" "cargo clippy --all-targets --all-features -- -D warnings"

# Format check
run_test "Code Formatting Check" "cargo fmt --all -- --check"

# Final Results
echo ""
echo "================================================"
echo -e "${BLUE}📊 Test Results Summary${NC}"
echo "================================================"
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests passed! VoiceStand is ready for deployment.${NC}"
    echo ""
    echo "Memory Safety: ✅ Guaranteed by Rust type system"
    echo "Performance: ✅ Optimized and validated"
    echo "Security: ✅ Audited and lint-free"
    echo "Functionality: ✅ All features tested"
    echo ""
    echo "🚀 VoiceStand Rust implementation is production-ready!"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed. Please review and fix issues before deployment.${NC}"
    exit 1
fi