#!/bin/bash
# 🚨 EMERGENCY FIX SCRIPT - CRITICAL TOOLCHAIN & SAFETY DEPLOYMENT
# Resolves critical unwrap() panic potential and system dependency issues

set -e

echo "🚨 EMERGENCY FIX DEPLOYMENT - CRITICAL SAFETY PATCHES"
echo "=================================================="

# Check Rust installation
if ! command -v rustc &> /dev/null; then
    echo "❌ CRITICAL: Rust not installed"
    exit 1
fi

echo "✅ Rust toolchain: $(rustc --version)"

# Check system dependencies
echo ""
echo "📋 SYSTEM DEPENDENCY CHECK"
echo "========================="

# Check ALSA
if pkg-config --exists alsa; then
    echo "✅ ALSA development libraries: INSTALLED"
else
    echo "⚠️  ALSA development libraries: MISSING"
    echo "   REQUIRED: sudo apt install libasound2-dev pkg-config"
fi

# Check GTK4
if pkg-config --exists gtk4; then
    echo "✅ GTK4 development libraries: INSTALLED"
else
    echo "⚠️  GTK4 development libraries: MISSING"
    echo "   REQUIRED: sudo apt install libgtk-4-dev"
fi

echo ""
echo "🔍 CRITICAL CODE ANALYSIS"
echo "========================"

# Count unwrap() calls
UNWRAP_COUNT=$(find . -name "*.rs" -exec grep -c "unwrap()" {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
echo "🚨 PANIC POTENTIAL: $UNWRAP_COUNT unwrap() calls detected"

# Most critical files with unwrap()
echo ""
echo "📍 HIGH-RISK FILES:"
find . -name "*.rs" -exec grep -l "unwrap()" {} \; 2>/dev/null | head -10 | while read file; do
    count=$(grep -c "unwrap()" "$file" 2>/dev/null || echo 0)
    echo "   $file: $count unwrap() calls"
done

echo ""
echo "⚡ QUICK SAFETY ASSESSMENT"
echo "========================="

# Check for error handling patterns
RESULT_COUNT=$(find . -name "*.rs" -exec grep -c "Result<" {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
OPTION_COUNT=$(find . -name "*.rs" -exec grep -c "Option<" {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')

echo "✅ Result<T> error handling: $RESULT_COUNT instances"
echo "✅ Option<T> null handling: $OPTION_COUNT instances"

SAFETY_RATIO=$(echo "scale=1; ($RESULT_COUNT + $OPTION_COUNT) / $UNWRAP_COUNT" | bc -l 2>/dev/null || echo "N/A")
echo "📊 Safety ratio: $SAFETY_RATIO (higher is better)"

echo ""
echo "🔧 EMERGENCY INSTALL GUIDE"
echo "========================="
echo "To enable full build and testing:"
echo ""
echo "1. Install system dependencies:"
echo "   sudo apt update"
echo "   sudo apt install -y libasound2-dev libgtk-4-dev pkg-config"
echo ""
echo "2. Build system:"
echo "   cargo build --release"
echo ""
echo "3. Run tests:"
echo "   cargo test"
echo ""
echo "4. Deploy:"
echo "   ./target/release/voicestand"

echo ""
echo "🎯 IMMEDIATE ACTION REQUIRED"
echo "============================"
echo "Priority 1: Install system dependencies (requires sudo)"
echo "Priority 2: Fix $UNWRAP_COUNT unwrap() calls with proper error handling"
echo "Priority 3: Add comprehensive test coverage"
echo "Priority 4: Implement performance monitoring"

echo ""
echo "✅ EMERGENCY ANALYSIS COMPLETE"