import RPi.GPIO as GPIO
from time import sleep

# Set the GPIO mode
GPIO.setmode(GPIO.BOARD)

# Define the GPIO pin for the servo
servo_pin = 18

# Set the PWM frequency and duty cycle range
frequency = 50  # Hz
duty_cycle_range = 100  # 0-100%

# Initialize the servo GPIO pin
GPIO.setup(servo_pin, GPIO.OUT)
servo_pwm = GPIO.PWM(servo_pin, frequency)

# Define the calibration value
calibrate = 7

def right(angle):
    angle = 90 + angle
    set_servo_angle(angle)

def left(angle):
    angle = 90 - angle
    set_servo_angle(angle)

def reset():
    set_servo_angle(90 + calibrate)

def turn(deg):
    angle = 90 + deg
    set_servo_angle(angle)

def set_servo_angle(angle):
    if angle > 180:
        angle = 180
    elif angle < 0:
        angle = 0
    duty_cycle = angle / 180 * duty_cycle_range
    servo_pwm.start(duty_cycle)
    sleep(0.5)  # Adjust the sleep time as needed
    servo_pwm.stop()