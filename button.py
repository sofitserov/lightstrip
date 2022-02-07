import sys
import RPi.GPIO as GPIO
import time
import config
import visualization

def green_callback(channel):
    visualization.start_energy()
    return


def red_callback(channel):
    visualization.stop_everything()
    return


def setup_buttons():
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(config.BUTTON_GREEN_PIN,
               GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(config.BUTTON_RED_PIN,
               GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(config.BUTTON_GREEN_PIN,
                      GPIO.RISING, callback=green_callback,
                      bouncetime=200)
    GPIO.add_event_detect(config.BUTTON_RED_PIN,
                      GPIO.RISING, callback=red_callback,
                      bouncetime=200)
    return



