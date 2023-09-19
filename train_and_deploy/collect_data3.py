import evdev
import RPi.GPIO as GPIO
import cv2 as cv
from datetime import datetime
import asyncio
import os
import sys
from time import time
import csv

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

start_stamp = time()
ave_frame_rate = 0.
start_time=datetime.now().strftime("%Y_%m_%d_%H_%M_")


device_path = '/dev/input/event0'  # controller

# create data storage
image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y_%m_%d_%H_%M"), 'images/')
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')

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

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

async def handle_input_events(device):
    global steer, throttle, action

    for event in device.async_read_loop():
        if event.type == evdev.ecodes.EV_ABS:
            if event.code == 0:  # X-axis of the left joystick (servo control)
                steer = event.value
            elif event.code == 5:  # Y-axis of the right joystick (motor control)
                throttle = event.value

async def control_servo_and_motor():
    global steer, throttle, action

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

async def log_data():
    global steer, throttle, action

    while True:
        frame = cv.resize(frame, (120, 160))
        cv.imwrite(image_dir + str(frame_counts)+'.jpg', frame) # changed frame to gray
        # save labels
        label = [start_time+str(frame_counts)+'.jpg'] + action
        with open(label_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(label)  # write the data
        # monitor frame rate
        duration_since_start = time() - start_stamp
        ave_frame_rate = frame_counts / duration_since_start

async def main():
    device = evdev.InputDevice(device_path)
    print(f"Reading input events from {device.name}...")

    input_task = asyncio.create_task(handle_input_events(device))
    control_task = asyncio.create_task(control_servo_and_motor())
    log_data = asyncio.create_task(log_data())

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
