from adafruit_servokit import ServoKit
import pygame
import os
#import servo
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# AN1 = 26
# DIG1 = 19
PWM_PIN = 26
DIR_PIN = 19

GPIO.setup(PWM_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)

# p1 = GPIO.PWM(AN1, 100)
pwm = GPIO.PWM(PWM_PIN, 1000)
pwm.start(0)

#import this to run with pygame and no webcam
os.environ["SDL_VIDEODRIVER"] = "dummy"

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
        if throttle > 0:
            GPIO.output(DIR_PIN, GPIO.LOW)  # forward
            pwm.ChangeDutyCycle(throttle)
        elif throttle < 0:
            GPIO.output(DIR_PIN, GPIO.HIGH)  # backward
            throttle = -throttle
            pwm.ChangeDutyCycle(throttle)
        else:
            pwm.ChangeDutyCycle(0)
                #steer = -js.get_axis(3)  # steer_input: -1: left, 1: right
       
        # ang = 90 * (1 + steer)
        # if ang > 180:
        #     ang = 180
        # elif ang < 0:
        #     ang = 0
        # servo.angle = ang
        
except KeyboardInterrupt:
    motor.kill()
    pygame.quit()
