"""
Integrated test with servo, motor, camera, controller.
Add data collection function to achieve the next step. 
"""
import sys
import cv2 as cv
import pygame
from gpiozero import Servo, PhaseEnableMotor
from time import time


# SETUP
# Load configs
THROTTLE_AXIS = 1
STEER_AXIS = 2
STEER_CENTER = 0.2
STEER_OFFSET = 0.7
STEER_DIR = -1  # 1: steer left if steer.value > 0; -1: steer left if steer.value < 0. 
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
                if js.get_button(11):  # start button 
                    throttle.stop()
                    throttle.close()
                    steer.close()
                    cv.destroyAllWindows()
                    pygame.quit()
                    print("E-STOP PRESSED. TERMINATE")
                    sys.exit()
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
        # Log action
        action = [act_st, act_th]
        print(f"action: {action}")
        # Log frame rate
        frame_counts += 1
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