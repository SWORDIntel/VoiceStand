# Mouse Button Discovery Guide

This guide helps you discover and map all the extra buttons on your mouse for use with VoiceStand.

## Quick Start

### 1. Build the Monitor Tool

```bash
# Quick build (standalone)
./build_mouse_monitor.sh

# OR via CMake (full build)
cd build
make mouse-button-monitor
```

### 2. Run the Monitor

```bash
./mouse-button-monitor
```

### 3. Test Your Mouse Buttons

Press each button on your mouse while watching the output. The monitor will show:

- **Button Number**: X11 button ID
- **Description**: Human-readable name
- **Binding String**: Ready-to-use configuration string
- **Modifiers**: Any keys held (Ctrl, Alt, Shift, etc.)
- **Raw Data**: Debug information

## Example Output

```
=== VoiceStand Mouse Button Monitor ===
Time        Event           Button   Description               Modifiers           Coordinates
-----------------------------------------------------------------------------------------------
14:23:45    PRESS           8        Back/Mouse4 (Button8)     None                (1920,540)
    → Binding string: "Back"
    → Raw: button=8 state=0x0

14:23:47    PRESS           10       Extra1 (Button10)         None                (1920,540)
    → Binding string: "Button10"
    → Raw: button=10 state=0x0

14:23:52    PRESS           3        Right (Button3)           Ctrl                (1920,540)
    → Binding string: "Ctrl+Right"
    → Raw: button=3 state=0x4
```

## Common Mouse Layouts

### Gaming Mice (Common Extra Buttons)

| Button Location | Typical Button # | VoiceStand Name | Binding String |
|-----------------|------------------|-----------------|----------------|
| Side (back) | 8 | Back | `"Back"` |
| Side (forward) | 9 | Forward | `"Forward"` |
| DPI/Profile | 10-11 | Extra1, Extra2 | `"Button10"`, `"Button11"` |
| Additional thumb | 12-15 | Extra3-6 | `"Button12"` to `"Button15"` |
| Scroll tilt left | 6 | ScrollLeft | `"ScrollLeft"` |
| Scroll tilt right | 7 | ScrollRight | `"ScrollRight"` |

### Office Mice

| Button Location | Typical Button # | VoiceStand Name | Binding String |
|-----------------|------------------|-----------------|----------------|
| Side (back) | 8 | Back | `"Back"` |
| Side (forward) | 9 | Forward | `"Forward"` |
| Scroll tilt | 6, 7 | ScrollLeft/Right | `"ScrollLeft"`, `"ScrollRight"` |

### Advanced Gaming Mice (12+ buttons)

Many gaming mice have 12-17 buttons. Use the monitor to discover them all:

```bash
# Run monitor and systematically press each button
./mouse-button-monitor

# Common ranges:
# Buttons 1-5: Standard (L, M, R, ScrollUp, ScrollDown)
# Buttons 6-7: Horizontal scroll
# Buttons 8-9: Side navigation
# Buttons 10-17: Programmable buttons
```

## Testing Procedure

### 1. Systematic Button Discovery

1. **Start the monitor**: `./mouse-button-monitor`
2. **Test standard buttons**:
   - Left click → Should show Button 1
   - Right click → Should show Button 3
   - Middle click → Should show Button 2
   - Scroll up/down → Should show Button 4/5

3. **Test extra buttons**:
   - Side buttons (usually thumb area)
   - Top buttons (often DPI/profile switches)
   - Additional programmable buttons
   - Scroll wheel tilt (if available)

4. **Test with modifiers**:
   - Hold Ctrl and press each extra button
   - Hold Alt and press each extra button
   - Try combinations: Ctrl+Alt+Button

### 2. Record Your Findings

Create a mapping table for your mouse:

```
My Mouse: [Brand/Model]
=======================
Physical Location    → Button # → VoiceStand Binding
Left thumb (back)    → 8        → "Back"
Left thumb (forward) → 9        → "Forward"
Top button 1         → 10       → "Button10"
Top button 2         → 11       → "Button11"
```

## Configuration Examples

### Gaming Mouse Setup

```json
{
  "hotkeys": {
    "toggle_recording": "Ctrl+Alt+Space",
    "toggle_recording_mouse": "Back",
    "push_to_talk": "Forward",
    "quick_transcribe": "Button10",
    "pause_recording": "Ctrl+Button11",
    "volume_up": "ScrollLeft",
    "volume_down": "ScrollRight"
  }
}
```

### Office Mouse Setup

```json
{
  "hotkeys": {
    "toggle_recording": "Ctrl+Alt+Space",
    "toggle_recording_mouse": "Ctrl+Back",
    "push_to_talk": "Ctrl+Forward"
  }
}
```

### Advanced Gaming Setup (Multi-button)

```json
{
  "hotkeys": {
    "toggle_recording": "Back",
    "push_to_talk": "Forward",
    "pause_recording": "Button10",
    "save_transcription": "Button11",
    "clear_text": "Button12",
    "settings": "Ctrl+Button13",
    "volume_up": "Button14",
    "volume_down": "Button15"
  }
}
```

## Troubleshooting

### Button Not Detected

1. **Check mouse drivers**: Some mice require specific drivers
2. **Test with other software**: Verify button works in other applications
3. **Check X11 mapping**: Use `xinput test <device-id>`
4. **Update mouse firmware**: Some mice need firmware updates

### Wrong Button Numbers

1. **Mouse-specific mapping**: Different mice map buttons differently
2. **Driver interference**: Gaming mouse software may remap buttons
3. **X11 configuration**: Check if xorg.conf modifies button mapping

### Modifiers Not Working

1. **Clean modifiers**: Monitor shows both raw and clean modifier states
2. **Desktop environment**: Some DEs intercept certain combinations
3. **Application conflicts**: Other apps may be grabbing the same combinations

## Advanced Discovery

### Using xinput

```bash
# List input devices
xinput list

# Monitor specific device (replace <id> with your mouse ID)
xinput test <id>

# Check device properties
xinput list-props <id>
```

### Using evtest (if available)

```bash
# Install evtest
sudo apt install evtest  # Ubuntu/Debian

# Test device (as root)
sudo evtest /dev/input/eventX
```

## Supported Button Range

VoiceStand supports X11 mouse buttons 1-31:

- **Buttons 1-5**: Standard mouse buttons + scroll
- **Buttons 6-7**: Horizontal scroll (if supported)
- **Buttons 8-9**: Common side buttons (back/forward)
- **Buttons 10-31**: Extended buttons (device-specific)

## Integration with VoiceStand

### Method 1: Direct Configuration

Edit your config file directly:

```bash
# Edit main config
nano ~/.config/voice-to-text/config.json

# Add mouse button bindings
{
  "hotkeys": {
    "toggle_recording_mouse": "Back"
  }
}
```

### Method 2: Command Line

```bash
# Test a specific button binding
./voice-to-text --test-binding "Button10"

# Register mouse button at runtime (if implemented)
./voice-to-text --add-mouse-binding "Back" --action "toggle_recording"
```

## Best Practices

### Safe Button Choices

✅ **Recommended**:
- Side buttons (Button8, Button9) - rarely used by other apps
- Extra buttons (Button10+) - usually available
- Modified combinations (Ctrl+Button8) - very safe

⚠️ **Use with caution**:
- ScrollLeft/ScrollRight - some apps use these
- DPI buttons - may be needed for mouse sensitivity

❌ **Avoid**:
- Standard buttons (Left, Right, Middle) without modifiers
- ScrollUp/ScrollDown - breaks normal scrolling

### Ergonomic Considerations

1. **Frequently used actions**: Assign to easily reachable buttons
2. **Emergency stops**: Use easily accessible buttons
3. **Modifier combos**: For less frequent actions
4. **Thumb buttons**: Great for primary voice control actions

## Debugging

### Enable Debug Mode

The monitor tool provides debug information:

```bash
# Run with all debug info
./mouse-button-monitor 2>&1 | tee mouse-debug.log

# Check for patterns in button numbers
grep "button=" mouse-debug.log | sort | uniq -c
```

### Common Issues and Solutions

1. **Buttons above 15 not detected**:
   - Some mice only support buttons 1-15
   - Check mouse specifications

2. **Inconsistent button numbers**:
   - Restart X11 session
   - Check for conflicting mouse software
   - Try different USB ports

3. **Modifiers not clean**:
   - NumLock/CapsLock can interfere
   - The monitor shows clean vs. raw modifiers

Ready to discover your mouse's capabilities? Run `./mouse-button-monitor` and start pressing buttons!