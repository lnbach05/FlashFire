import pygame
from gpiozero import AngularServo

pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)

servo = AngularServo(17, min_angle=-90, max_angle=90)

while True:
    #Main program
    for e in pygame.event.get():
        if e.type == pygame.JOYAXISMOTION:
            #Get joystick steeing angle value from joystick and save it to a variable 
            steer = -js.get_axis(2)  # steer_input: -1: left, 1: right
    ang = -35 + steer * 55
    servo.angle = ang