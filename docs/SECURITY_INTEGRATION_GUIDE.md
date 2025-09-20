# Security Interface Integration Guide

## Quick Integration

To integrate the Adaptive Security Interface with VoiceStand:

### 1. Add to CMakeLists.txt

```cmake
# Add security interface sources
set(SECURITY_SOURCES
    src/gui/security_interface.cpp
    src/gui/security_hardware_bridge.cpp
    # Optional: src/gui/security_dialogs.cpp
)

target_sources(voice-to-text PRIVATE ${SECURITY_SOURCES})

# Link additional libraries
target_link_libraries(voice-to-text PRIVATE dl)  # For dynamic library loading
```

### 2. Update Build Dependencies

Ensure these dependencies are available:

```bash
# Required for hardware detection
sudo apt-get install libtss2-dev        # TPM support
sudo apt-get install intel-media-driver # Intel ME/NPU support
sudo apt-get install libssl-dev         # Cryptographic support
```

### 3. Build the Rust Security Library

```bash
# Build the Rust security modules as a shared library
cd src/
cargo build --release --lib --crate-type=cdylib
cp target/release/libvoicestand_security.so ../
```

### 4. Runtime Configuration

The security interface will:
- Automatically detect available hardware
- Gracefully degrade if hardware is unavailable
- Store configuration in `~/.config/voice-to-text/security.json`

## Testing the Integration

### Quick Test

```bash
# Build and run
./build.sh
./build/voice-to-text

# Look for security initialization messages:
# "Security interface initialized successfully"
# "Security level: X detected"
```

### Hardware Detection Test

```bash
# Check available security hardware
ls /dev/tpm* /dev/mei* /dev/accel* 2>/dev/null

# Verify TPM functionality
sudo tpm2_getcap properties-fixed 2>/dev/null

# Check Intel ME status
sudo cat /sys/kernel/debug/mei/*/devstate 2>/dev/null
```

## Security Levels

The interface automatically adapts to these levels:

| Level | Hardware | UI Behavior |
|-------|----------|-------------|
| 1 - Minimal | None | No security UI visible |
| 2 - Basic | NPU | Basic security indicator |
| 3 - Good | TPM + NPU | Full security panel |
| 4 - Optimal | TPM + ME + NPU + GNA | Enterprise features |

## Customization

### Hiding Security Features

```cpp
// Disable security interface entirely
// In main_window.cpp, comment out:
// initialize_security_interface();
```

### Enterprise Configuration

```json
// ~/.config/voice-to-text/security.json
{
  "version": "1.0",
  "auto_detect_hardware": true,
  "enable_enterprise_features": true,
  "metrics_update_interval": 1000,
  "enterprise": {
    "policy_enforcement_enabled": true,
    "audit_retention_days": 90,
    "automatic_key_rotation": true
  }
}
```

### Custom Hardware Detection

```cpp
// In security_hardware_bridge.cpp, modify:
bool SecurityHardwareBridge::detect_custom_hardware() {
    // Add custom hardware detection logic
    return check_custom_security_device();
}
```

## Troubleshooting

### Common Issues

#### Security Interface Not Visible
- Check hardware detection: No security hardware found
- Solution: Interface only shows with Level 2+ security

#### Hardware Detection Fails
- Check permissions: User needs access to security devices
- Solution: Add user to `tss` group for TPM access

#### Performance Issues
- Check metrics update frequency in configuration
- Solution: Increase `metrics_update_interval` to 2000ms or higher

#### Rust Library Loading Fails
- Check library path and permissions
- Solution: Ensure `libvoicestand_security.so` is in library path

### Debug Mode

```cpp
// Enable debug output in security_interface.cpp
#define SECURITY_DEBUG 1

// Or set environment variable:
export VOICESTAND_SECURITY_DEBUG=1
```

## Enterprise Deployment

### Policy Configuration

```bash
# Deploy enterprise policy
sudo mkdir -p /etc/voicestand/
sudo cp enterprise-policy.json /etc/voicestand/security-policy.json

# Set system-wide configuration
export VOICESTAND_ENTERPRISE_MODE=1
export VOICESTAND_POLICY_SERVER=https://policy.company.com
```

### Compliance Features

When enterprise features are enabled:

- Automatic audit logging to system journal
- Policy enforcement for encryption settings
- Compliance dashboard for administrators
- Certificate management integration

### Multi-User Support

```bash
# System-wide configuration
sudo mkdir -p /etc/voicestand/
sudo chown root:voicestand /etc/voicestand/
sudo chmod 750 /etc/voicestand/

# User group for VoiceStand users
sudo groupadd voicestand
sudo usermod -a -G voicestand,tss $USER
```

## API Reference

### Main Classes

```cpp
// Core security interface
SecurityInterface security_interface;
security_interface.initialize(parent_window);
security_interface.integrate_with_main_window(main_box, status_bar);

// Hardware detection bridge
SecurityHardwareBridge bridge;
auto capabilities = bridge.detect_hardware_capabilities();
auto metrics = bridge.get_current_security_metrics();

// Configuration management
security_interface.set_security_config_callback(config_callback);
security_interface.set_hardware_detection_callback(detection_callback);
```

### Key Methods

```cpp
// Hardware capabilities
HardwareCapabilities caps = bridge.detect_hardware_capabilities();
bool has_tpm = caps.tpm_available;
SecurityLevel level = caps.current_level;

// Real-time metrics
SecurityMetrics metrics = bridge.get_current_security_metrics();
uint64_t operations = metrics.operations_completed;
uint32_t latency = metrics.current_latency_ms;

// Configuration
bridge.configure_security_level(SecurityLevel::OPTIMAL);
bridge.enable_enterprise_features(true);
bridge.set_encryption_algorithm("AES-256-GCM");
```

## Performance Tuning

### Optimize for Different Use Cases

```cpp
// High performance: Reduce update frequency
security_config_["metrics_update_interval"] = 5000;  // 5 seconds

// Real-time monitoring: Increase update frequency
security_config_["metrics_update_interval"] = 250;   // 250ms

// Enterprise: Enable full logging
security_config_["enterprise"]["detailed_logging"] = true;

// Embedded: Minimal UI
security_config_["ui_mode"] = "minimal";
```

### Memory Usage

- Basic mode: ~50KB additional memory
- Full mode: ~200KB additional memory
- Enterprise mode: ~500KB additional memory

### CPU Usage

- Background monitoring: <1% CPU
- Active metrics: 1-2% CPU
- Hardware operations: 2-5% CPU

## Support and Maintenance

### Logging

Security events are logged to:

```bash
# Application logs
~/.config/voice-to-text/voicestand.log

# System security logs (enterprise mode)
/var/log/voicestand-security.log

# Audit logs (enterprise mode)
~/.config/voice-to-text/audit_logs.json
```

### Updates

The security interface is designed for forward compatibility:

- Configuration format versioning
- Graceful handling of new hardware types
- Automatic migration of settings
- Backward compatibility with older configurations

### Contributing

When contributing to the security interface:

1. Maintain graceful degradation
2. Add comprehensive error handling
3. Update documentation and tooltips
4. Test on systems without security hardware
5. Follow secure coding practices
6. Add appropriate audit logging

---

For detailed architecture information, see [ADAPTIVE_SECURITY_INTERFACE.md](ADAPTIVE_SECURITY_INTERFACE.md).