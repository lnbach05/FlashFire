from gpiozero import Motor, Servo
from inputs import get_gamepad
import time

# Define the motor and servo GPIO pins
motor = Motor(forward=19, backward=26)  # Replace with your actual GPIO pin
servo = Servo(24)  # Example GPIO pin, adjust to your setup

# Function to map joystick values to motor/servo control
def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

try:
    print("Waiting for game controller input...")
    while True:
        events = get_gamepad()

        for event in events:
            if event.ev_type == "Absolute":
                if event.ev_type == "ABS_X":  # Joystick X-axis (motor control)
                    motor_speed = map_value(event.ev_value, -32768, 32767, -1, 1)
                    motor.value = motor_speed

                elif event.ev_type == "ABS_Y":  # Joystick Y-axis (servo control)
                    servo_position = map_value(event.ev_value, -32768, 32767, -1, 1)
                    servo.value = servo_position

except KeyboardInterrupt:
    pass
finally:
    motor.stop()
    servo.value = None
