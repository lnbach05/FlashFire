import evdev
import RPi.GPIO as GPIO
import cv2 as cv
from time import time
from datetime import datetime
import sys
import os
import json
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

servo_range = 100 #0-100%

device_path = '/dev/input/event0' #controller

# create data storage
image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y_%m_%d_%H_%M"), 'images/')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')

#init variables
is_recording = True
frame_counts = 0

# init camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not initialzed!!")

cap.set(cv.CAP_PROP_FPS, 20)
for i in reversed(range(60)):  # warm up camera
    if not i % 20:
        print(i/20)
    ret, frame = cap.read()
# init timer, uncomment if you are cuious about frame rate
start_stamp = time()
ave_frame_rate = 0.
start_time=datetime.now().strftime("%Y_%m_%d_%H_%M_")

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

try:
    device = evdev.InputDevice(device_path)
    print(f"Reading input events from {device.name}...")

    for event in device.read_loop(): #Continuous loop

        ret, frame = cap.read()
        if frame is not None:
            frame_counts += 1
            print(frame_counts)
    
        
            if event.type == evdev.ecodes.EV_ABS:
                if event.code == 0: #X-axis of the left joystick (servo control)
                    axis_event = evdev.ecodes.ABS[event.code]
                    steer = event.value
                    servo_angle = float(map_range(steer, 0, 255, 7.7, 11.7)) #turning
                    servo_pwm.ChangeDutyCycle(servo_angle)
                    print(f'Steer: {steer}')

        #             elif event.code == 5: #Y-axis of the right joystick (motor control)
        #                 axis_event = evdev.ecodes.ABS[event.code]
        #                 throttle = event.value

        #                 # Map the axis value to motor speed (0% to 100%)
        #                 speed = float(map_range(throttle, 128, 0, 0, 80))
        #                 if speed < 0:
        #                     motor_pwm.ChangeDutyCycle(0)
        #                 else:
        #                     motor_pwm.ChangeDutyCycle(speed)

        #action = [steer, throttle]

        if is_recording:
            frame = cv.resize(frame, (120, 160))
            cv.imwrite(image_dir + str(frame_counts)+'.jpg', frame) # changed frame to gray
            # save labels
            # label = [start_time+str(frame_counts)+'.jpg'] + action
            # with open(label_path, 'a+', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(label)  # write the data
        # monitor frame rate
        duration_since_start = time() - start_stamp
        ave_frame_rate = frame_counts / duration_since_start

except FileNotFoundError:
    print(f"Device not found at {device_path}")
except KeyboardInterrupt:
    pass
finally:
    motor_pwm.stop()
    GPIO.cleanup()
