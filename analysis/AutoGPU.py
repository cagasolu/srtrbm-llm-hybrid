from pynvml import *
import time


class GPUEnergyTracker:
    """
    GPU energy monitor using NVIDIA NVML.

    Measures instantaneous power and integrates
    it over time to estimate total GPU energy
    consumption.
    Units
    Power  : Watts
    Energy : Joules
    """

    def __init__(self, gpu_index=0):
        nvmlInit()

        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)

        self.energy_joules = 0.0

        self.last_time = time.time()

        self.last_power = self._read_power()

    def _read_power(self):
        power_mw = nvmlDeviceGetPowerUsage(self.handle)

        return power_mw / 1000.0

    def step(self):
        current_time = time.time()

        power = self._read_power()

        dt = current_time - self.last_time

        # trapezoidal integration
        self.energy_joules += 0.5 * (self.last_power + power) * dt

        self.last_power = power
        self.last_time = current_time

    def total_energy(self):
        return self.energy_joules

    def current_power(self):
        return self.last_power