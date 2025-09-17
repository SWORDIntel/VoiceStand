#!/bin/bash

# Personal GNA Week 1 Build Script for Dell Latitude 5450 MIL-SPEC
# Focus: Personal productivity system validation

echo "=== Personal GNA Week 1 Build System ==="
echo "Target: Dell Latitude 5450 MIL-SPEC"
echo "Focus: Personal push-to-talk excellence"

# Create build directory
mkdir -p build_personal_gna
cd build_personal_gna

echo ""
echo "Building Personal GNA Components..."

# Build personal GNA device manager test
echo "  [1/4] Compiling Personal GNA Device Manager..."
g++ -std=c++17 -O2 -march=native -mavx2 \
    -I../src \
    -Wall -Wextra \
    -DPERSONAL_GNA_BUILD \
    -c ../src/core/gna_device_manager.cpp \
    -o gna_device_manager.o

if [ $? -eq 0 ]; then
    echo "    ✅ Personal GNA Device Manager compiled successfully"
else
    echo "    ❌ Personal GNA Device Manager compilation failed"
    exit 1
fi

# Build personal GNA voice detector
echo "  [2/4] Compiling Personal GNA Voice Detector..."
g++ -std=c++17 -O2 -march=native -mavx2 \
    -I../src \
    -Wall -Wextra \
    -DPERSONAL_GNA_BUILD \
    -c ../src/core/gna_voice_detector.cpp \
    -o gna_voice_detector.o

if [ $? -eq 0 ]; then
    echo "    ✅ Personal GNA Voice Detector compiled successfully"
else
    echo "    ❌ Personal GNA Voice Detector compilation failed"
    exit 1
fi

# Build personal integration test framework
echo "  [3/4] Compiling Personal Integration Test Framework..."
g++ -std=c++17 -O2 -march=native -mavx2 \
    -I../src \
    -Wall -Wextra \
    -DPERSONAL_GNA_BUILD \
    -c ../src/test/personal_gna_integration_test.cpp \
    -o personal_gna_integration_test.o

if [ $? -eq 0 ]; then
    echo "    ✅ Personal Integration Test Framework compiled successfully"
else
    echo "    ❌ Personal Integration Test Framework compilation failed"
    exit 1
fi

# Build personal Week 1 demo
echo "  [4/4] Compiling Personal Week 1 Demo..."
g++ -std=c++17 -O2 -march=native -mavx2 \
    -I../src \
    -Wall -Wextra \
    -DPERSONAL_GNA_BUILD \
    ../src/integration/personal_gna_week1_demo.cpp \
    gna_device_manager.o \
    gna_voice_detector.o \
    personal_gna_integration_test.o \
    -lpthread -lm \
    -o personal_gna_week1_demo

if [ $? -eq 0 ]; then
    echo "    ✅ Personal Week 1 Demo compiled successfully"
else
    echo "    ❌ Personal Week 1 Demo compilation failed"
    exit 1
fi

echo ""
echo "=== Personal GNA Week 1 Build Complete ==="
echo "Executable: ./build_personal_gna/personal_gna_week1_demo"
echo ""
echo "To run Personal Week 1 validation:"
echo "  cd build_personal_gna"
echo "  ./personal_gna_week1_demo"
echo ""
echo "Personal GNA Integration Ready for Testing!"