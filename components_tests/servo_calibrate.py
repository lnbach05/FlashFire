from gpiozero import Servo

servo = Servo(17)

try:
    while True:
        servo_value = float(input("Please enter a value between -1 and 1: "))
        servo.value = servo_value
except KeyboardInterrupt:
    print("Done")