import os
import sys
import json
import pygame
from pygame.locals import JOYAXISMOTION
from pygame import event
import motor

config_path = os.path.join(sys.path[0], "config.json")
f = open(config_path)
data = json.load(f)
throttle_lim = data['throttle_lim']

pygame.joystick.init()
js = pygame.joystick.Joystick(0)
throttle = 0.

try:
    while True:
        throttle = -round((js.get_axis(1)), 2)  # throttle input: -1: max forward, 1: max backward
        motor.drive(throttle * throttle_lim)  # apply throttle limit

except KeyboardInterrupt:
    motor.kill()
    pygame.quit()
    sys.exit()


