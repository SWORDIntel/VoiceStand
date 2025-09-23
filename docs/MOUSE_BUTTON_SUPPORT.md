# Mouse Button Support for VoiceStand

VoiceStand now supports mouse button triggers in addition to keyboard hotkeys, providing more flexible and ergonomic control options for voice recognition.

## Features

### Supported Mouse Buttons

| Button Name | Aliases | X11 Button | Description |
|-------------|---------|------------|-------------|
| `Left` | `Button1`, `LeftClick` | 1 | Left mouse button |
| `Middle` | `Button2`, `MiddleClick` | 2 | Middle mouse button (scroll wheel click) |
| `Right` | `Button3`, `RightClick` | 3 | Right mouse button |
| `ScrollUp` | `Button4`, `WheelUp` | 4 | Scroll wheel up |
| `ScrollDown` | `Button5`, `WheelDown` | 5 | Scroll wheel down |
| `Back` | `Button8`, `Mouse4` | 8 | Side button (back) |
| `Forward` | `Button9`, `Mouse5` | 9 | Side button (forward) |

### Modifier Keys

Mouse buttons can be combined with modifier keys:

- `Ctrl` or `Control`
- `Alt`
- `Shift`
- `Super`, `Win`, or `Meta`

### Examples

```bash
# Basic mouse buttons
Right                   # Right click
Middle                  # Middle click
Back                    # Side button (back)

# With modifiers
Ctrl+Right             # Ctrl + right click
Alt+Middle             # Alt + middle click
Shift+ScrollUp         # Shift + scroll wheel up
Ctrl+Alt+Back          # Ctrl + Alt + side button

# Numeric format
Button3                # Same as "Right"
Ctrl+Button8           # Same as "Ctrl+Back"
```

## Configuration

### JSON Configuration

Add mouse button bindings to your configuration file:

```json
{
  "hotkeys": {
    "toggle_recording": "Ctrl+Alt+Space",
    "toggle_recording_mouse": "Right",
    "push_to_talk": "Ctrl+Alt+V",
    "additional_bindings": {
      "quick_transcribe": "Ctrl+Middle",
      "pause_recording": "Alt+Right",
      "volume_up": "Ctrl+ScrollUp",
      "volume_down": "Ctrl+ScrollDown"
    }
  }
}
```

### Programmatic Usage

```cpp
#include "integration/hotkey_manager.h"

vtt::HotkeyManager manager;
manager.initialize();

// Register mouse button
manager.register_mouse_button("Ctrl+Right");

// Register keyboard hotkey (still supported)
manager.register_hotkey("Ctrl+Alt+Space");

// Set callback for both types
manager.set_hotkey_callback([](const std::string& binding) {
    std::cout << "Triggered: " << binding << std::endl;
});

manager.start();
```

## Implementation Details

### X11 Integration

Mouse button support is implemented using X11's `XGrabButton()` function:

- **Global Capture**: Mouse buttons are captured system-wide, even when VoiceStand isn't focused
- **Modifier Support**: Handles NumLock and CapsLock variations automatically
- **Event Processing**: Processes `ButtonPress` events in the main event loop

### Architecture

```
HotkeyManager
├── InputBinding struct      # Unified keyboard/mouse binding storage
├── register_mouse_button()  # Mouse button registration
├── parse_mouse_button_string() # Parse button + modifier strings
└── event_loop()            # Handle both KeyPress and ButtonPress events
```

### Backward Compatibility

- Existing keyboard hotkey functionality remains unchanged
- Configuration files without mouse buttons continue to work
- Legacy `register_hotkey()` method still supported

## Security Considerations

⚠️ **Important Security Notes:**

1. **Global Hotkeys**: Mouse button bindings are global system hotkeys that work across all applications
2. **Application Interference**: Be careful not to choose bindings that conflict with other applications
3. **Accessibility**: Consider users who may rely on specific mouse button functions
4. **Permission Requirements**: Requires X11 access to capture global mouse events

### Recommended Safe Bindings

```bash
# Generally safe combinations
Ctrl+Right             # Rarely used by applications
Alt+Middle             # Safe for most applications
Shift+ScrollUp         # Usually available
Ctrl+Back              # Side buttons often unused
```

### Avoid These Bindings

```bash
# Commonly used by applications
Left                   # Basic selection/interaction
Right                  # Context menus
Middle                 # Paste in Linux
ScrollUp/ScrollDown    # Standard scrolling
```

## Testing and Discovery

### Mouse Button Discovery Tool

**First, discover what buttons your mouse has:**

```bash
# Quick build and run
./build_mouse_monitor.sh
./mouse-button-monitor
```

This tool will:
1. Show real-time button numbers as you press them
2. Display suggested binding strings for configuration
3. Show modifier key combinations
4. Help you map all extra buttons on your mouse

### Test Program

After discovering your buttons, test them with VoiceStand:

```bash
cd build
make test-mouse-buttons
./test-mouse-buttons
```

The test program will:
1. Register various mouse button combinations
2. Display triggered bindings in real-time
3. Help you verify your mouse button configuration

**See [MOUSE_BUTTON_DISCOVERY.md](MOUSE_BUTTON_DISCOVERY.md) for complete discovery guide.**

### Manual Testing

1. Configure a mouse button binding in VoiceStand
2. Start VoiceStand
3. Try the configured mouse button combination
4. Verify the action triggers (e.g., recording starts/stops)

## Troubleshooting

### Common Issues

**Mouse buttons not working:**
- Check if you have the required X11 permissions
- Verify your mouse has the buttons you're trying to bind
- Test with the `test-mouse-buttons` program first

**Conflicts with other applications:**
- Choose less common button combinations
- Use modifier keys to avoid conflicts
- Check system-wide hotkey managers (like desktop environments)

**Side buttons not detected:**
- Ensure your mouse driver supports side buttons
- Check `xinput list` to see if buttons are recognized
- Some mice require specific drivers or configuration

### Debug Commands

```bash
# List input devices
xinput list

# Monitor mouse events
xinput test <device-id>

# Check X11 button mapping
xmodmap -pp
```

## Performance

- **Latency**: Mouse button detection adds minimal latency (~1ms)
- **CPU Usage**: Negligible impact on system performance
- **Memory**: Small increase in memory usage for button tracking
- **Battery**: No significant impact on battery life

## Future Enhancements

Planned improvements:
- [ ] Mouse gesture support (drag patterns)
- [ ] Double-click detection
- [ ] Configurable button hold duration
- [ ] Per-application mouse button bindings
- [ ] GUI configuration interface for mouse buttons

## Examples

See `examples/mouse_button_config.json` for a complete configuration example with various mouse button bindings.