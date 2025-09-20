# VoiceStand GUI Architecture

## Overview

VoiceStand features a GTK4-based graphical user interface with adaptive security features that appear based on hardware detection. The GUI provides both basic voice-to-text functionality and optional enterprise-grade security controls.

## Core GUI Components

### Main Window (`main_window.cpp/h`)
The primary application window built with GTK4, featuring:

```cpp
class MainWindow {
private:
    GtkWidget* window;
    GtkWidget* main_box;           // Vertical container
    GtkWidget* control_panel;      // Push-to-talk controls
    GtkWidget* text_display;       // Transcription output
    GtkWidget* status_bar;         // System status
    GtkWidget* security_panel;     // Conditional security interface

public:
    void initialize();
    void setup_push_to_talk();
    void setup_text_display();
    void setup_security_interface();
};
```

## Adaptive Security Interface

### Hardware Detection Levels

The GUI adapts based on detected Intel hardware capabilities:

#### Level 1: Basic (Software Only)
- **Hardware**: No specialized security hardware
- **Visible**: Standard voice-to-text interface only
- **Features**: Basic transcription, settings
- **Security Panel**: Hidden

#### Level 2: Enhanced (NPU Available)
- **Hardware**: Intel NPU detected
- **Visible**: Basic status indicator
- **Features**: NPU acceleration indicator
- **Security Panel**: Minimal status only

#### Level 3: Secure (NPU + TPM)
- **Hardware**: NPU + TPM 2.0 detected
- **Visible**: Security dashboard
- **Features**: Hardware crypto status, security metrics
- **Security Panel**: Full security interface

#### Level 4: Enterprise (Full Stack)
- **Hardware**: NPU + TPM + Intel ME + GNA
- **Visible**: Complete enterprise interface
- **Features**: All security controls, compliance dashboard
- **Security Panel**: Full enterprise features

## GUI Layout Structure

### Main Interface Layout
```
┌─────────────────────────────────────────────────┐
│ VoiceStand - Intel Hardware Accelerated VTT    │
├─────────────────────────────────────────────────┤
│ ┌─ Push-to-Talk Controls ─────────────────────┐ │
│ │ [🎤 Activate] [⚙️ Settings] [📊 Status]    │ │
│ │ Hotkey: Ctrl+Space | GNA: "Hey VoiceStand"  │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─ Transcription Display ─────────────────────┐ │
│ │                                             │ │
│ │ [Real-time transcription appears here...]   │ │
│ │                                             │ │
│ │ [Auto-scrolling text with speaker ID]      │ │
│ │                                             │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─ Security Panel (Conditional) ──────────────┐ │
│ │ 🔒 Security Level: [Detected Level]         │ │
│ │ Hardware: [✓ NPU] [✓ TPM] [○ ME] [✓ GNA]   │ │
│ │ Encryption: AES-256-GCM (Hardware)          │ │
│ │ [Security Settings] [Compliance Dashboard]  │ │
│ └─────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────┤
│ Status: Ready | NPU: Active | Latency: 2.8ms   │
└─────────────────────────────────────────────────┘
```

## Component Functionality

### Push-to-Talk Controls
```cpp
class PushToTalkPanel {
private:
    GtkWidget* activate_button;    // Manual activation
    GtkWidget* settings_button;    // Configuration
    GtkWidget* status_indicator;   // Current state

public:
    void setup_hotkey_detection();      // Ctrl+Space binding
    void setup_gna_integration();       // "Hey VoiceStand" wake word
    void update_activation_state();     // Visual feedback
    void configure_activation_mode();   // Dual/single mode
};
```

### Transcription Display
```cpp
class TranscriptionDisplay {
private:
    GtkWidget* text_view;          // Scrollable text
    GtkTextBuffer* buffer;         // Text content
    GtkWidget* speaker_labels;     // Speaker identification

public:
    void append_transcription(const std::string& text);
    void add_speaker_marker(int speaker_id);
    void highlight_keywords();
    void export_text();
    void clear_display();
};
```

### Security Interface (Adaptive)
```cpp
class SecurityInterface {
private:
    SecurityLevel detected_level;
    GtkWidget* security_expander;   // Collapsible panel
    GtkWidget* hardware_status;     // Component indicators
    GtkWidget* metrics_display;     // Performance metrics
    GtkWidget* settings_dialog;     // Security configuration

public:
    void detect_hardware_capabilities();
    void adapt_ui_to_capabilities();
    void show_hardware_status();
    void display_security_metrics();
    void open_security_settings();
};
```

## Security Interface Details

### Hardware Status Indicators
```cpp
struct HardwareIndicators {
    bool npu_available;      // ✓ Intel NPU (11 TOPS)
    bool tpm_available;      // ✓ TPM 2.0 (Hardware Crypto)
    bool me_available;       // ○ Intel ME (Ring -3)
    bool gna_available;      // ✓ GNA (Wake Word)
};

void SecurityInterface::update_indicators() {
    gtk_label_set_text(npu_label,
        npu_available ? "✓ NPU (Active)" : "○ NPU (Unavailable)");
    gtk_label_set_text(tpm_label,
        tmp_available ? "✓ TPM 2.0 (Ready)" : "○ TPM (Missing)");
    // ... other indicators
}
```

### Security Metrics Display
```cpp
struct SecurityMetrics {
    std::string encryption_algorithm;  // "AES-256-GCM"
    std::string key_source;           // "TPM Hardware"
    double crypto_latency_ms;         // 0.2ms
    uint64_t operations_completed;    // 1,247,892
    bool compliance_status;           // FIPS 140-2 compliant
};

void SecurityInterface::display_metrics() {
    gtk_label_set_text(encryption_label,
        format("Encryption: {} ({})", metrics.encryption_algorithm, metrics.key_source));
    gtk_label_set_text(latency_label,
        format("Crypto Latency: {:.1f}ms", metrics.crypto_latency_ms));
    // ... other metrics
}
```

## Configuration Dialogs

### Security Settings Dialog
- **Encryption Algorithm Selection**: AES-256-GCM, ChaCha20-Poly1305
- **Key Management**: TPM-backed, Software fallback
- **Compliance Mode**: FIPS 140-2, Common Criteria
- **Audit Logging**: Enable/disable security event logging

### Audio Settings Dialog
- **Input Device**: Microphone selection
- **Sample Rate**: 16kHz, 44.1kHz, 48kHz
- **Noise Reduction**: Enable/disable
- **VAD Sensitivity**: Energy threshold adjustment

### Activation Settings Dialog
- **Hotkey Configuration**: Customizable key combinations
- **GNA Wake Word**: Custom phrase training
- **Activation Mode**: Push-to-talk, Voice command, Both
- **Timeout Settings**: Auto-deactivation timing

## Event Handling

### Audio Events
```cpp
void MainWindow::on_audio_detected() {
    // Update status indicator
    gtk_widget_set_sensitive(activate_button, TRUE);
    // Start transcription display updates
}

void MainWindow::on_transcription_ready(const std::string& text) {
    // Append to text display
    transcription_display.append_transcription(text);
    // Update word count and statistics
}
```

### Security Events
```cpp
void SecurityInterface::on_hardware_change() {
    // Re-detect capabilities
    detect_hardware_capabilities();
    // Adapt UI elements
    adapt_ui_to_capabilities();
    // Update status indicators
}

void SecurityInterface::on_security_event(SecurityEvent event) {
    // Log security events
    // Update metrics display
    // Show notifications if configured
}
```

## Accessibility Features

### Keyboard Navigation
- Full keyboard accessibility for all controls
- Tab navigation through interface elements
- Keyboard shortcuts for all major functions

### Screen Reader Support
- Proper ARIA labels for all components
- Status announcements for transcription updates
- Security state announcements

### Visual Accessibility
- High contrast mode support
- Customizable font sizes
- Color-blind friendly indicators

## Integration Points

### Rust Backend Integration
```cpp
class RustBackendBridge {
public:
    void initialize_audio_pipeline();
    void start_voice_detection();
    void get_transcription_update();
    void check_security_status();
    void configure_hardware_settings();
};
```

### Hardware Detection Bridge
```cpp
class HardwareDetectionBridge {
public:
    SecurityLevel detect_security_level();
    HardwareCapabilities get_capabilities();
    SecurityMetrics get_current_metrics();
    bool configure_security_level(SecurityLevel target);
};
```

## Build Integration

### GTK4 Dependencies
```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK4 REQUIRED gtk4)
pkg_check_modules(GLIB REQUIRED glib-2.0)

target_link_libraries(voicestand-gui
    ${GTK4_LIBRARIES}
    ${GLIB_LIBRARIES}
    voicestand-core
)
```

### Resource Management
- Automatic cleanup of GTK widgets
- RAII patterns for resource handling
- Proper signal disconnection on shutdown

## Performance Considerations

### Real-time Updates
- Non-blocking transcription display updates
- Efficient text buffer management
- Minimal CPU usage for GUI updates

### Memory Management
- Bounded text buffer size
- Automatic old text cleanup
- Efficient widget update patterns

### Responsive Design
- Adaptive layout for different window sizes
- Collapsible panels for space efficiency
- Configurable interface density

## Future Enhancements

### Planned Features
- **Theme Support**: Dark/light mode adaptation
- **Multi-monitor**: Floating transcription windows
- **Customization**: User-configurable layouts
- **Plugins**: Extension interface for additional features

### Enterprise Features
- **Compliance Dashboard**: Real-time compliance monitoring
- **Audit Viewer**: Security event log browser
- **Policy Management**: Enterprise policy enforcement
- **Remote Management**: Centralized configuration

This GUI architecture provides a foundation that scales from basic voice-to-text functionality to enterprise-grade secure voice processing, adapting automatically to available hardware capabilities.