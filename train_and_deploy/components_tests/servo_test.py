import  RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

GPIO.setup(11, GPIO.OUT)
servo1 = GPIO.PWM(11, 50)

servo1.start(0)
print("Waiting")
time.sleep(2)

print("Rotating 180 degrees")

duty = 2

while duty <= 12:
    servo1.ChangeDutyCycle(duty)
    time.sleep(1)
    duty = duty + 1

time.sleep(2)

print("Turning back to 90 degrees")
servo1.ChangeDutyCycle(7)
time.sleep(2)

print("Turning back to 0 degrees")
servo1.ChangeDutyCycle(2)
time.sleep(0.5)
servo1.ChangeDutyCycle(0)

servo1.stop()
GPIO.cleanup()
print("Done")


