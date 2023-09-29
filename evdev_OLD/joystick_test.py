import evdev

device_path = '/dev/input/event0'  # Modify this with your controller's device path

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

try:
    device = evdev.InputDevice(device_path)
    print(f"Reading input events from {device.name}...")
    while True:
        for event in device.read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                if event.code == 0:
                    left_event = evdev.ecodes.ABS[event.code]
                    left_value = event.value

                    print(f"Axis Value: {left_value}")
                elif event.code == 5:
                    right_event = evdev.ecodes.ABS[event.code]
                    right_value = event.value

                    print(f"Axis Value: {right_value}")

            elif event.type == evdev.ecodes.EV_KEY:
                button_event = evdev.ecodes.EV_KEY[event.code]
                button_value = event.value

                #print(f"Button Value: "{button_value})

except FileNotFoundError:
    print(f"Device not found at {device_path}")
except KeyboardInterrupt:
    pass



