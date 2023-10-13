from gpiozero import Servo

servo = Servo(24)

try:
    while True:
        servo_value = input("Please enter a value between -1 and 1: ")
        servo.value = servo_value
except KeyboardInterrupt:
    print("Done")