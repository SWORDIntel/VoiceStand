#!/bin/bash
# 🔍 DEPLOYMENT VALIDATION - NO SUDO REQUIRED
# Validates emergency deployment progress and identifies remaining blockers

set -e

echo "🔍 VOICESTAND RUST DEPLOYMENT VALIDATION"
echo "========================================"

# Source Rust environment
source ~/.cargo/env 2>/dev/null || {
    echo "❌ CRITICAL: Rust environment not available"
    exit 1
}

echo "✅ Rust Environment: $(rustc --version)"

# Check codebase structure
echo ""
echo "📁 CODEBASE STRUCTURE VALIDATION"
echo "================================"

EXPECTED_DIRS=("voicestand" "voicestand-core" "voicestand-audio" "voicestand-speech" "voicestand-gui")
for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        lines=$(find "$dir" -name "*.rs" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
        echo "✅ $dir: $lines lines of Rust code"
    else
        echo "❌ Missing: $dir"
    fi
done

# Safety analysis
echo ""
echo "🚨 SAFETY CRITICAL ANALYSIS"
echo "=========================="

TOTAL_UNWRAPS=0
for file in $(find . -name "*.rs" 2>/dev/null); do
    count=$(grep -c "unwrap()" "$file" 2>/dev/null || echo "0")
    if [ "$count" -gt 0 ] 2>/dev/null; then
        TOTAL_UNWRAPS=$((TOTAL_UNWRAPS + count))
        echo "⚠️  $file: $count unwrap() calls"
    fi
done

echo ""
echo "📊 SAFETY METRICS:"
echo "   Total unwrap() calls: $TOTAL_UNWRAPS"

if [ "$TOTAL_UNWRAPS" -eq 0 ]; then
    echo "   ✅ EXCELLENT: No panic potential detected"
elif [ "$TOTAL_UNWRAPS" -lt 10 ]; then
    echo "   🟡 GOOD: Low panic potential ($TOTAL_UNWRAPS calls)"
elif [ "$TOTAL_UNWRAPS" -lt 30 ]; then
    echo "   ⚠️  WARNING: Moderate panic potential ($TOTAL_UNWRAPS calls)"
else
    echo "   🚨 CRITICAL: High panic potential ($TOTAL_UNWRAPS calls)"
fi

# Dependency check
echo ""
echo "📦 SYSTEM DEPENDENCY STATUS"
echo "=========================="

# Check ALSA
if pkg-config --exists alsa 2>/dev/null; then
    echo "✅ ALSA development libraries: Available"
    ALSA_OK=1
else
    echo "❌ ALSA development libraries: Missing"
    echo "   Install with: sudo apt install libasound2-dev"
    ALSA_OK=0
fi

# Check GTK4
if pkg-config --exists gtk4 2>/dev/null; then
    echo "✅ GTK4 development libraries: Available"
    GTK_OK=1
else
    echo "❌ GTK4 development libraries: Missing"
    echo "   Install with: sudo apt install libgtk-4-dev"
    GTK_OK=0
fi

# Build attempt (will fail without dependencies)
echo ""
echo "🔨 BUILD VALIDATION"
echo "==================="

if [ "$ALSA_OK" -eq 1 ] && [ "$GTK_OK" -eq 1 ]; then
    echo "🔨 Attempting full build..."
    if cargo check --all-targets 2>/dev/null; then
        echo "✅ BUILD SUCCESS: All components compile"
        BUILD_OK=1
    else
        echo "❌ BUILD FAILED: Compilation errors detected"
        echo "   Run 'cargo build' for detailed error information"
        BUILD_OK=0
    fi
else
    echo "⚠️  SKIPPING BUILD: System dependencies missing"
    echo "   Install dependencies first, then run build validation"
    BUILD_OK=0
fi

# Final assessment
echo ""
echo "📋 DEPLOYMENT ASSESSMENT"
echo "======================="

SCORE=0

if [ "$TOTAL_UNWRAPS" -lt 10 ]; then SCORE=$((SCORE + 25)); fi
if [ "$ALSA_OK" -eq 1 ]; then SCORE=$((SCORE + 25)); fi
if [ "$GTK_OK" -eq 1 ]; then SCORE=$((SCORE + 25)); fi
if [ "$BUILD_OK" -eq 1 ]; then SCORE=$((SCORE + 25)); fi

echo "🎯 READINESS SCORE: $SCORE/100"

if [ "$SCORE" -ge 75 ]; then
    echo "✅ STATUS: PRODUCTION READY"
    echo "   System is ready for deployment and testing"
elif [ "$SCORE" -ge 50 ]; then
    echo "🟡 STATUS: PARTIALLY READY"
    echo "   Some issues remain but core functionality available"
elif [ "$SCORE" -ge 25 ]; then
    echo "⚠️  STATUS: NEEDS WORK"
    echo "   Significant issues prevent reliable operation"
else
    echo "🚨 STATUS: NOT READY"
    echo "   Critical issues must be resolved before use"
fi

echo ""
echo "🎯 NEXT ACTIONS REQUIRED:"

if [ "$ALSA_OK" -eq 0 ] || [ "$GTK_OK" -eq 0 ]; then
    echo "1. Install system dependencies (requires sudo access)"
fi

if [ "$TOTAL_UNWRAPS" -gt 5 ]; then
    echo "2. Fix remaining $TOTAL_UNWRAPS unwrap() calls for safety"
fi

if [ "$BUILD_OK" -eq 0 ] && [ "$ALSA_OK" -eq 1 ] && [ "$GTK_OK" -eq 1 ]; then
    echo "3. Debug compilation errors"
fi

echo ""
echo "🔍 VALIDATION COMPLETE - Run with dependencies installed for full assessment"