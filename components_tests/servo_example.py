from gpiozero import Servo
from time import sleep

servo = Servo(17)

try:
    for _ in range(2):  # loop twice
        servo.min()
        sleep(0.5)
        servo.mid()
        sleep(0.5)
        servo.max()
        sleep(0.5)
except KeyboardInterrupt:
    print("Stopped")

