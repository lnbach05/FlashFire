
# Collect training data using bluetooth controller
# While driving, save images and joystick inputs
import sys
import os
import cv2 as cv
from gpiozero import PhaseEnableMotor
from gpiozero import Servo
import pygame
import csv
from datetime import datetime

from time import time


# SETUP
# dummy video driver
os.environ["SDL_VIDEODRIVER"] = "dummy"
image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y_%m_%d_%H_%M"), 'images/')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')

#Initialize controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

#Initialize variables
throttle, steer = 0., 0.
is_recording = True
frame_counts = 0

#Initialize camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 20)
start_stamp = time()
ave_frame_rate = 0.
start_time=datetime.now().strftime("%Y_%m_%d_%H_%M_")

#Initialize motor and servo objects
motor = PhaseEnableMotor(phase=19, enable=26)
servo = Servo(17)
CENTER = 0.05
OFFSET = 0.5
servo.value = CENTER

# MAIN
try:
    while True:
        ret, frame = cap.read()
        if frame is not None:
            frame_counts += 1
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                throttle = (-js.get_axis(1)) * 0.5 
                steer = -js.get_axis(2)
            elif e.type == pygame.JOYBUTTONDOWN:
                print("Button pressed:", event.button)
                if e.button == 11: # Button START
                    pygame.quit()
                    raise SystemExit
                elif e.button == 0:  # Button Y
                    if not is_recording:
                        is_recording = True
                        print("Recording started")
                        
                elif e.button == 2:  # Button A
                    if is_recording:
                        is_recording = False
                        print("Recording stopped")

        if throttle > 0:
            motor.forward(throttle)
        elif throttle < 0:
            motor.backward(-throttle)

        if steer == 0:
            servo.value = CENTER
        elif steer > CENTER + OFFSET:
            servo.value = OFFSET
        elif steer < -(CENTER + OFFSET):
            servo.value = -OFFSET
        else:
            servo.value = steer

        action = [steer, throttle]
        
        if is_recording:
            frame = cv.resize(frame, (120, 160))
            # changed frame to gray
            cv.imwrite(image_dir + str(frame_counts)+'.jpg', frame) 
            # save labels
            label = [str(frame_counts)+'.jpg'] + action
            with open(label_path, 'a+', newline='') as f:
                # write the data
                writer = csv.writer(f)
                writer.writerow(label)  

        # monitor frame rate
        duration_since_start = time() - start_stamp
        ave_frame_rate = frame_counts / duration_since_start

#Command to exit script using control C        
except KeyboardInterrupt:
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()
