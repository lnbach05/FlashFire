import sys
import os
import cv2 as cv
import pygame
from gpiozero import Servo, PhaseEnableMotor
from time import time
import torch
from torchvision import transforms
import convnets


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
# Load configs and init servo controller
model_path = os.path.join(sys.path[0], 'models', model_name)
to_tensor = transforms.ToTensor()
model = convnets.DonkeyNet()  
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# init variables
throttle, steer = 0., 0.
is_recording = False
frame_counts = 0

# init camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 20)
for i in reversed(range(60)):  # warm up camera
    if not i % 20:
        print(i/20)
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

# MAIN
try:
    while True:
        ret, frame = cap.read()  # read image
        if not ret:
            print("No frame received. TERMINATE!")
            sys.exit()
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(11):  # START button 
                    throttle.stop()
                    throttle.close()
                    steer.close()
                    cv.destroyAllWindows()
                    pygame.quit()
                    print("E-STOP PRESSED. TERMINATE")
                    sys.exit()
        # predict steer and throttle
        image = cv.resize(frame, (120, 160))
        img_tensor = to_tensor(image)
        st_pred, th_pred = model(img_tensor[None, :]).squeeze()
        st_trim = float(st_pred)
        if st_trim >= 1:  # trim steering signal
            st_trim = .999
        elif st_trim <= -1:
            st_trim = -.999
        th_trim = (float(th_pred))
        if th_trim >= 1:  # trim throttle signal
            th_trim = .999
        elif th_trim <= -1:
            th_trim = -.999
        throttle.forward(throttle)
        steer.value = STEER_CENTER + st_trim * STEER_OFFSET * STEER_DIR
        print(f"predicted action: {st_pred, th_pred}")        
        frame_counts += 1
        # Log frame rate
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")
        if cv.waitKey(1)==ord('q'):
            cv.destroyAllWindows()
            sys.exit()
except KeyboardInterrupt:
    cv.destroyAllWindows()
    sys.exit()
