"""
Servo test script.
Also calibrates the servo.
Please Log your calibrated values below:
    center: 0.2
    offset: 0.7
So, you'll want to turn servo in a range of (center +/- offset) 
"""
from gpiozero import Servo
from time import sleep


servo = Servo(pin=17)  

# Observe CENTER and BOUNDARY while executing following loop
for v in range(-5, 10): 
    servo.value = v * 0.1
    print(f"servo value: {v * 0.1}")
    sleep(.5)
servo.value = 0.2
print("CENTER")
sleep(1)