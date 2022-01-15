import RPi.GPIO as GPIO


class Servo:
    def __init__(self, servo_pin=18) -> None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(servo_pin, GPIO.OUT)
        self.servo = GPIO.PWM(servo_pin, 50)
        self.servo.start(0)

    def set_angle(self, angle: int) -> None:
        angle = min(angle, 90)
        angle = max(angle, -90)
        self.servo.ChangeDutyCycle(2.5 + (12.0 - 2.5) * (angle + 90) / 180)

    def __del__(self) -> None:
        self.servo.stop()
        GPIO.cleanup()
