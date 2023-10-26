import pygame
import os
#import servo
from gpiozero import PhaseEnableMotor, Servo

#import this to run with pygame and no webcam
os.environ["SDL_VIDEODRIVER"] = "dummy"

# init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# init variables
throttle, steer = 0., 0.

motor = PhaseEnableMotor(phase=19, enable=26)
servo = Servo(24)

# MAIN
try:
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                throttle = -js.get_axis(1)  # throttle input: -1: max forward, 1: max backward
                steer = -js.get_axis(2)  # steer_input: -1: left, 1: right

        if throttle > 0:
            motor.forward(throttle)
        elif throttle < 0:
            motor.backward(-throttle)

        servo.value = steer

except KeyboardInterrupt:
    pygame.quit()
