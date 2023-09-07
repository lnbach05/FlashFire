import evdev
import RPi.GPIO as GPIO

# GPIO pins for motor control
motor_pwm_pin = 26 
motor_direction_pin1 = 19
servo_pin = 24

# Motor initialization
GPIO.setmode(GPIO.BCM)
GPIO.setup(motor_pwm_pin, GPIO.OUT)
GPIO.setup(motor_direction_pin1, GPIO.OUT)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM setup for motor speed control
motor_pwm = GPIO.PWM(motor_pwm_pin, 100)  # 100 Hz PWM frequency
motor_pwm.start(0)  # Start with 0% duty cycle (stopped)

servo_pwm = GPIO.PWM(servo_pin, 50)
servo_pwm.start(0)

device_path = '/dev/input/event0' #controller

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

try:
    device = evdev.InputDevice(device_path)
    print(f"Reading input events from {device.name}...")
    while True:
        for event in device.read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                if event.code == 0: #X-axis of the left joystick (servo control)
                    axis_event = evdev.ecodes.ABS[event.code]
                    axis_value = event.value
                    servo_angle = int(map_range(event.value, 0, 255, 0, 180)) #turning
                    servo_pwm.ChangeDutyCycle(servo_angle/10.0 + 2.5)

                elif event.code == 1: #Y-axis of the right joystick (motor control)
                    axis_event = evdev.ecodes.ABS[event.code]
                    axis_value = event.value

                    # Map the axis value to motor speed (0% to 100%)
                    speed = int(map_range(axis_value, 255, 0, -80, 80))
                    if speed < 0:
                        motor_pwm.ChangeDutyCycle(0)
                    else:
                        motor_pwm.ChangeDutyCycle(speed)

except FileNotFoundError:
    print(f"Device not found at {device_path}")
except KeyboardInterrupt:
    pass
finally:
    motor_pwm.stop()
    GPIO.cleanup()
