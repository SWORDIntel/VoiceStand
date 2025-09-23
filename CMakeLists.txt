cmake_minimum_required(VERSION 3.16)
project(VoiceToText VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GTK4 REQUIRED gtk4)
pkg_check_modules(PULSE REQUIRED libpulse)

find_package(X11 REQUIRED)
find_package(Threads REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(PkgConfig REQUIRED)

# Find libcurl for HTTP client
pkg_check_modules(CURL REQUIRED libcurl)

# Find FFTW3 for audio feature extraction (optional for basic functionality)
pkg_check_modules(FFTW3 fftw3)

# Check if FFTW3 was found
if(FFTW3_FOUND)
    message(STATUS "FFTW3 found - enabling advanced audio feature extraction")
    add_definitions(-DHAVE_FFTW3)
else()
    message(STATUS "FFTW3 not found - using basic audio features only")
endif()

set(WHISPER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/include" 
    CACHE PATH "Path to whisper.cpp include directory")
set(WHISPER_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/build" 
    CACHE PATH "Path to whisper.cpp library directory")

add_executable(voice-to-text
    src/main.cpp
    src/core/audio_capture.cpp
    src/core/whisper_processor.cpp
    src/core/personal_voice_commands.cpp
    src/core/learning_api_client.cpp
    src/core/ensemble_whisper_processor.cpp
    src/core/adaptive_learning_system.cpp
    src/core/audio_feature_serializer.cpp
    src/core/learning_system_manager.cpp
    src/core/settings_manager.cpp
    src/gui/main_window.cpp
    src/integration/hotkey_manager.cpp
)

target_include_directories(voice-to-text PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GTK4_INCLUDE_DIRS}
    ${PULSE_INCLUDE_DIRS}
    ${X11_INCLUDE_DIR}
    ${WHISPER_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
    ${CURL_INCLUDE_DIRS}
)

# Conditionally add FFTW3 includes
if(FFTW3_FOUND)
    target_include_directories(voice-to-text PRIVATE ${FFTW3_INCLUDE_DIRS})
endif()

target_compile_options(voice-to-text PRIVATE
    ${GTK4_CFLAGS_OTHER}
    ${PULSE_CFLAGS_OTHER}
    ${CURL_CFLAGS_OTHER}
    -Wall -Wextra -O2
)

# Conditionally add FFTW3 compile options
if(FFTW3_FOUND)
    target_compile_options(voice-to-text PRIVATE ${FFTW3_CFLAGS_OTHER})
endif()

target_link_directories(voice-to-text PRIVATE
    ${WHISPER_LIB_DIR}/src
    ${WHISPER_LIB_DIR}/ggml/src
)

target_link_libraries(voice-to-text PRIVATE
    ${GTK4_LIBRARIES}
    ${PULSE_LIBRARIES}
    ${X11_LIBRARIES}
    ${CURL_LIBRARIES}
    Xtst
    ${CMAKE_THREAD_LIBS_INIT}
    jsoncpp_lib
    whisper
    ggml
    m
    stdc++fs
)

# Conditionally add FFTW3 libraries
if(FFTW3_FOUND)
    target_link_libraries(voice-to-text PRIVATE ${FFTW3_LIBRARIES})
endif()

install(TARGETS voice-to-text
    RUNTIME DESTINATION bin
)

install(FILES resources/voice-to-text.desktop
    DESTINATION share/applications
)

install(FILES resources/voice-to-text.svg
    DESTINATION share/icons/hicolor/scalable/apps
)

add_custom_target(download-models
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/models
    COMMAND wget -nc -P ${CMAKE_CURRENT_SOURCE_DIR}/models 
            https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
    COMMENT "Downloading Whisper base model..."
)

# Personal GNA Integration - Week 1 Components
add_executable(personal_gna_week1_demo
    src/integration/personal_gna_week1_demo.cpp
    src/core/gna_device_manager.cpp
    src/core/gna_voice_detector.cpp
    src/test/personal_gna_integration_test.cpp
)

# GNA Voice Detector test executable
add_executable(gna_power_test
    src/core/gna_power_test.cpp
    src/core/gna_voice_detector.cpp
    src/core/gna_device_manager.cpp
)

# Check for OpenVINO installation
find_path(OPENVINO_INCLUDE_DIR
    NAMES openvino/openvino.hpp
    PATHS
        /home/john/openvino/runtime/include
        /opt/openvino/runtime/include
        /usr/local/include
        /usr/include
)

find_library(OPENVINO_LIBRARIES
    NAMES openvino
    PATHS
        /home/john/openvino/runtime/lib
        /opt/openvino/runtime/lib
        /usr/local/lib
        /usr/lib
)

if(OPENVINO_INCLUDE_DIR AND OPENVINO_LIBRARIES)
    message(STATUS "OpenVINO found - enabling GNA support")
    target_compile_definitions(personal_gna_week1_demo PRIVATE ENABLE_OPENVINO)
    target_include_directories(personal_gna_week1_demo PRIVATE ${OPENVINO_INCLUDE_DIR})
    target_link_libraries(personal_gna_week1_demo PRIVATE ${OPENVINO_LIBRARIES})

    target_compile_definitions(gna_power_test PRIVATE ENABLE_OPENVINO)
    target_include_directories(gna_power_test PRIVATE ${OPENVINO_INCLUDE_DIR})
    target_link_libraries(gna_power_test PRIVATE ${OPENVINO_LIBRARIES})
else()
    message(STATUS "OpenVINO not found - using software fallback")
endif()

# Personal GNA Week 1 Demo configuration
target_include_directories(personal_gna_week1_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(personal_gna_week1_demo PRIVATE
    -Wall -Wextra -O2 -march=native -mavx2
)

target_link_libraries(personal_gna_week1_demo PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
    m
    stdc++fs
)

# GNA Power Test configuration
target_include_directories(gna_power_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(gna_power_test PRIVATE
    -Wall -Wextra -O2 -march=native
)

target_link_libraries(gna_power_test PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
    m
    stdc++fs
)

# Test executable for integrated system
add_executable(test_integrated
    src/test/test_integrated_system.cpp
    src/core/audio_capture.cpp
    src/core/whisper_processor.cpp
    src/core/enhanced_whisper_processor.cpp
    src/core/settings_manager.cpp
)

target_include_directories(test_integrated PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GTK4_INCLUDE_DIRS}
    ${PULSE_INCLUDE_DIRS}
    ${X11_INCLUDE_DIR}
    ${WHISPER_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

target_compile_options(test_integrated PRIVATE
    ${GTK4_CFLAGS_OTHER}
    ${PULSE_CFLAGS_OTHER}
    -Wall -Wextra -O2
)

target_link_directories(test_integrated PRIVATE
    ${WHISPER_LIB_DIR}/src
    ${WHISPER_LIB_DIR}/ggml/src
)

target_link_libraries(test_integrated PRIVATE
    ${GTK4_LIBRARIES}
    ${PULSE_LIBRARIES}
    ${X11_LIBRARIES}
    Xtst
    ${CMAKE_THREAD_LIBS_INIT}
    jsoncpp_lib
    whisper
    ggml
    m
    stdc++fs
)

# Phase 2 Personal Commands Test Executable
add_executable(test_personal_commands
    src/test/test_phase2_personal_commands.cpp
    src/core/personal_voice_commands.cpp
    src/core/audio_capture.cpp
    src/core/enhanced_whisper_processor.cpp
    src/core/settings_manager.cpp
)

target_include_directories(test_personal_commands PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GTK4_INCLUDE_DIRS}
    ${PULSE_INCLUDE_DIRS}
    ${X11_INCLUDE_DIR}
    ${WHISPER_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

target_compile_options(test_personal_commands PRIVATE
    ${GTK4_CFLAGS_OTHER}
    ${PULSE_CFLAGS_OTHER}
    -Wall -Wextra -O2 -march=native -mavx2
)

target_link_directories(test_personal_commands PRIVATE
    ${WHISPER_LIB_DIR}/src
    ${WHISPER_LIB_DIR}/ggml/src
)

target_link_libraries(test_personal_commands PRIVATE
    ${GTK4_LIBRARIES}
    ${PULSE_LIBRARIES}
    ${X11_LIBRARIES}
    Xtst
    ${CMAKE_THREAD_LIBS_INIT}
    jsoncpp_lib
    whisper
    ggml
    m
    stdc++fs
)

# Mouse button test executable
add_executable(test-mouse-buttons
    test_mouse_buttons.cpp
    src/integration/hotkey_manager.cpp
)

target_include_directories(test-mouse-buttons PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${X11_INCLUDE_DIR}
)

target_compile_options(test-mouse-buttons PRIVATE
    -Wall -Wextra -O2
)

target_link_libraries(test-mouse-buttons PRIVATE
    ${X11_LIBRARIES}
    Threads::Threads
)

# Mouse button discovery tool
add_executable(mouse-button-monitor
    mouse_button_monitor.cpp
)

target_include_directories(mouse-button-monitor PRIVATE
    ${X11_INCLUDE_DIR}
)

target_compile_options(mouse-button-monitor PRIVATE
    -Wall -Wextra -O2
)

target_link_libraries(mouse-button-monitor PRIVATE
    ${X11_LIBRARIES}
)