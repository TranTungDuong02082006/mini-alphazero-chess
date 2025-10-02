from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
import time

nvmlInit()
device_count = 1  # Change this to the number of GPUs you want to monitor

try:
    while True:
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU {i}: Memory {memory.used / 1024**2:.2f}MB/{memory.total / 1024**2:.2f}MB, Utilization {utilization.gpu}%")
        time.sleep(5)  # Adjust the interval as needed
finally:
    nvmlShutdown()
