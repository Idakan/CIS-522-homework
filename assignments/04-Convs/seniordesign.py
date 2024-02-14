import time
from w1thermsensor import W1ThermSensor

sensor = W1ThermSensor()

while True:
    temperature = sensor.get_temperature()
    print("The temperature is %s celsius" % temperature)
    if temperature > 80:
        print("You are not eligible to enter.")
    time.sleep(1)
