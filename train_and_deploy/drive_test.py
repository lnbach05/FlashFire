import evdev
import RPi.GPIO as GPIO

# GPIO pins for motor control
motor_pwm_pin = 18  # Replace with your PWM pin
motor_direction_pin1 = 23  # Replace with your direction pin
motor_direction_pin2 = 24  # Replace with your direction pin

# Motor initialization
GPIO.setmode(GPIO.BCM)
GPIO.setup(motor_pwm_pin, GPIO.OUT)
GPIO.setup(motor_direction_pin1, GPIO.OUT)
GPIO.setup(motor_direction_pin2, GPIO.OUT)

# PWM setup for motor speed control
motor_pwm = GPIO.PWM(motor_pwm_pin, 100)  # 100 Hz PWM frequency
motor_pwm.start(0)  # Start with 0% duty cycle (stopped)

# Replace '/dev/input/eventX' with the actual device path for your controller
device_path = '/dev/input/event0'  # Modify this with your controller's device path

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

try:
    device = evdev.InputDevice(device_path)
    print(f"Reading input events from {device.name}...")
    while True:
        for event in device.read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                axis_event = evdev.ecodes.ABS[event.code]
                axis_value = event.value

                # Map the axis value to motor speed (0% to 100%)
                speed = int(map_range(axis_value, 255, 0, -80, 80))
                motor_pwm.ChangeDutyCycle(speed)

            elif event.type == evdev.ecodes.EV_KEY:
                # Handle button presses/releases if needed
                pass

except FileNotFoundError:
    print(f"Device not found at {device_path}")
except KeyboardInterrupt:
    pass
finally:
    motor_pwm.stop()
    GPIO.cleanup()
