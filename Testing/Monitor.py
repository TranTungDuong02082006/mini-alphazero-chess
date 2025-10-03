import os
import time

while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    os.system('nvidia-smi')
    time.sleep(5)  # Adjust the interval as needed
