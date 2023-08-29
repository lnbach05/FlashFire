import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM) #GPIO Number

# Define the GPIO pin for the servo
servo_pin = 18

# Set the PWM frequency and duty cycle range
frequency = 50  # Hz
duty_cycle_range = 100  # 0-100%

# Initialize the servo GPIO pin
GPIO.setup(servo_pin, GPIO.OUT)
servo_pwm = GPIO.PWM(servo_pin, frequency)

# Function to set the servo angle
def set_servo_angle(angle):
    if angle > 180:
        angle = 180
    elif angle < 0:
        angle = 0
    duty_cycle = (angle / 180.0) * duty_cycle_range
    servo_pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(1)  # Adjust the sleep time as needed

# Start PWM
servo_pwm.start(0)  # Start with 0% duty cycle

try:
    set_servo_angle(180)  # left bound
    print("left bound: 180 deg")
    time.sleep(2)
    set_servo_angle(0)  # right bound
    print("right bound: 0 deg")
    time.sleep(2)
    set_servo_angle(90)  # middle
    print("middle: 90 deg")
    time.sleep(2)

except KeyboardInterrupt:
    servo_pwm.stop()
    GPIO.cleanup()
