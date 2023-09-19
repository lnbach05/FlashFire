import evdev
import RPi.GPIO as GPIO
import cv2 as cv
from datetime import datetime
import asyncio

# GPIO pins for motor control
motor_pwm_pin = 26
motor_direction_pin1 = 19
servo_pin = 24

# Motor initialization
GPIO.setmode(GPIO.BCM)
GPIO.setup(motor_pwm_pin, GPIO.OUT)
GPIO.setup(motor_direction_pin1, GPIO.OUT)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM setup for motor speed control
motor_pwm = GPIO.PWM(motor_pwm_pin, 100)  # 100 Hz PWM frequency
motor_pwm.start(0)  # Start with 0% duty cycle (stopped)

servo_pwm = GPIO.PWM(servo_pin, 50)
servo_pwm.start(0)

servo_range = 100  # 0-100%

device_path = '/dev/input/event0'  # controller

# create data storage
image_dir = 'your_image_directory'  # Replace with your image directory path
label_path = 'your_label_path'  # Replace with your label file path

# init variables
is_recording = True
frame_counts = 0

# init camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not initialized!!")

cap.set(cv.CAP_PROP_FPS, 20)
for i in reversed(range(60)):  # warm up camera
    if not i % 20:
        print(i / 20)
    ret, frame = cap.read()

# Initialize motor and servo values
steer = 0
throttle = 0

async def handle_input_events(device):
    global steer, throttle

    for event in device.async_read_loop():
        if event.type == evdev.ecodes.EV_ABS:
            if event.code == 0:  # X-axis of the left joystick (servo control)
                steer = event.value
            elif event.code == 5:  # Y-axis of the right joystick (motor control)
                throttle = event.value

async def control_servo_and_motor():
    global steer, throttle

    while True:
        # Control servo based on steer value
        servo_angle = float(map_range(steer, 0, 255, 7.7, 11.7))  # turning
        servo_pwm.ChangeDutyCycle(servo_angle)

        # Control motor based on throttle value
        speed = float(map_range(throttle, 128, 0, 0, 80))
        if speed < 0:
            motor_pwm.ChangeDutyCycle(0)
        else:
            motor_pwm.ChangeDutyCycle(speed)

        await asyncio.sleep(0.01)  # Adjust the sleep duration as needed

async def main():
    device = evdev.InputDevice(device_path)
    print(f"Reading input events from {device.name}...")

    input_task = asyncio.create_task(handle_input_events(device))
    control_task = asyncio.create_task(control_servo_and_motor())

    try:
        while True:
            await asyncio.sleep(1)  # Keep the main loop running
    except KeyboardInterrupt:
        pass
    finally:
        input_task.cancel()
        control_task.cancel()
        await input_task
        await control_task

if __name__ == "__main__":
    asyncio.run(main())
