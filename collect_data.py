import sys
import os
from datetime import datetime
import cv2 as cv
import pygame
from gpiozero import Servo, PhaseEnableMotor
from time import time
import csv


# SETUP
# Load configs
THROTTLE_AXIS = 1
STEER_AXIS = 2
STEER_CENTER = 0.2
STEER_OFFSET = 0.7
STEER_DIR = -1  # 1: steer left if steer.value < 0; -1: steer left if steer.value > 0. 
THROTTLE_LIMIT = 0.3
# Init servo 
steer = Servo(pin=17)
steer.value = STEER_CENTER #Starting angle
# Init motor 
throttle = PhaseEnableMotor(phase=19, enable=26)
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# Create data directory
image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y_%m_%d_%H_%M"), 'images/')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
# Init camera
cap = cv.VideoCapture(2)
cap.set(cv.CAP_PROP_FPS, 20)
for i in reversed(range(60)):
    ret, frame = cap.read()
    if not ret:
        print("No frame received. TERMINATE!")
        sys.exit()
    if not i % 20:
        print(i/20)  # count down 3, 2, 1 sec   
# Init timer for FPS computing
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.
# Init joystick axes values
st_ax_val, th_ax_val = 0., 0.
is_recording = False

# MAIN LOOP
try:
    while True:
        ret, frame = cap.read()  # read image
        if not ret:
            print("No frame received. TERMINATE!")
            sys.exit()
        # cv.imshow('camera', cv.resize(frame, (320, 240)))
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYAXISMOTION:
                st_ax_val = round((js.get_axis(STEER_AXIS)), 2)  
                th_ax_val = round((js.get_axis(THROTTLE_AXIS)), 2)  # keep 2 decimals
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(11):  # START button 
                    throttle.stop()
                    throttle.close()
                    steer.close()
                    cv.destroyAllWindows()
                    pygame.quit()
                    print("E-STOP PRESSED. TERMINATE")
                    sys.exit()
                elif js.get_button(0):  # A button
                    is_recording = not is_recording
        # Calaculate steering and throttle value
        act_st = st_ax_val  # steer_input: -1: left, 1: right
        act_th = -th_ax_val  # throttle input: -1: max forward, 1: max backward
        # Drive servo
        steer.value = STEER_CENTER + act_st * STEER_OFFSET * STEER_DIR
        # Drive motor
        if act_th >= 0.1:
            throttle.forward(min(act_th, THROTTLE_LIMIT))
        elif act_th <= -0.1:
            throttle.backward(min(-act_th, THROTTLE_LIMIT))
        else:
            throttle.stop()
        # Log data
        action = [act_st, act_th]
        print(f"action: {action}")
        if is_recording:
            img = cv.resize(frame, (120, 160))
            cv.imwrite(image_dir + str(frame_counts) + '.jpg', img)
            label = [str(frame_counts) + '.jpg'] + action
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)
        frame_counts += 1
        # Log frame rate
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")
        # Press "q" to quit
        if cv.waitKey(1)==ord('q'):
            throttle.stop()
            throttle.close()
            steer.close()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
            
# Take care terminate signal (Ctrl-c)
except KeyboardInterrupt:
    throttle.stop()
    throttle.close()
    steer.close()
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()