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

set(WHISPER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp" 
    CACHE PATH "Path to whisper.cpp include directory")
set(WHISPER_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/build" 
    CACHE PATH "Path to whisper.cpp library directory")

add_executable(voice-to-text
    src/main.cpp
    src/core/audio_capture.cpp
    src/core/whisper_processor.cpp
    src/gui/main_window.cpp
    src/integration/hotkey_manager.cpp
)

target_include_directories(voice-to-text PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GTK4_INCLUDE_DIRS}
    ${PULSE_INCLUDE_DIRS}
    ${X11_INCLUDE_DIR}
    ${WHISPER_INCLUDE_DIR}
)

target_compile_options(voice-to-text PRIVATE
    ${GTK4_CFLAGS_OTHER}
    ${PULSE_CFLAGS_OTHER}
    -Wall -Wextra -O2
)

target_link_directories(voice-to-text PRIVATE
    ${WHISPER_LIB_DIR}
)

target_link_libraries(voice-to-text PRIVATE
    ${GTK4_LIBRARIES}
    ${PULSE_LIBRARIES}
    ${X11_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    jsoncpp_lib
    whisper
    m
    stdc++fs
)

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