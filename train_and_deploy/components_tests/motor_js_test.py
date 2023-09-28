from adafruit_servokit import ServoKit
import pygame
import motor
import servo

# init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# init variables
throttle, steer = 0., 0.

# MAIN
try:
    while True:
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                throttle = -js.get_axis(1)  # throttle input: -1: max forward, 1: max backward
                steer = -js.get_axis(3)  # steer_input: -1: left, 1: right
        motor.drive(throttle)  # apply throttle limit
        ang = 90 * (1 + steer)
        if ang > 180:
            ang = 180
        elif ang < 0:
            ang = 0
        servo.angle = ang
        
except KeyboardInterrupt:
    motor.kill()
    pygame.quit()
