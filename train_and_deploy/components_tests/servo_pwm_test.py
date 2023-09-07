import evdev
import RPi.GPIO as GPIO

# GPIO pins for servo control
servo_pin = 24

# Motor initialization
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM setup for servo control
servo_pwm = GPIO.PWM(servo_pin, 50)
servo_pwm.start(0)

try:
    while True:
        pwm = float(input("Please enter a pwm value to test the servo: "))
        print(f"Turning servo with pwm = {pwm}")

        servo_pwm.ChangeDutyCycle(pwm)

except KeyboardInterrupt:
    pass

finally:
    servo_pwm.stop()