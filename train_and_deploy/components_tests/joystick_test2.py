import evdev

# Replace '/dev/input/eventX' with the actual device path for your controller
device_path = '/dev/input/event0'  # Modify this with your controller's device path

try:
    device = evdev.InputDevice(device_path)
    print(f"Reading input events from {device.name}...")

    for event in device.read_loop():
        if event.type == evdev.ecodes.EV_KEY:
            key_event = evdev.ecodes.KEY[event.code]
            key_state = "pressed" if event.value == 1 else "released"
            print(f"Key {key_event} {key_state}")

        elif event.type == evdev.ecodes.EV_ABS:
            axis_event = evdev.ecodes.ABS[event.code]
            axis_value = event.value
            print(f"Axis {axis_event} {axis_value}")

except FileNotFoundError:
    print(f"Device not found at {device_path}")
