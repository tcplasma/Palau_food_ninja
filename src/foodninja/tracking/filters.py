import math
import time

class LowPassFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.prev_val = None

    def filter(self, val: float) -> float:
        if self.prev_val is None:
            self.prev_val = val
            return val
        filtered = self.alpha * val + (1.0 - self.alpha) * self.prev_val
        self.prev_val = filtered
        return filtered

class OneEuroFilter:
    """
    Implementation of the One Euro Filter for reducing jitter in real-time tracking.
    Reference: http://www.casereview.org/publications/Casiez-CHI12.pdf
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_filt = LowPassFilter(self._alpha(min_cutoff, 1.0/60.0))
        self.dx_filt = LowPassFilter(self._alpha(d_cutoff, 1.0/60.0))
        
        self.prev_time = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, val: float, timestamp: float | None = None) -> float:
        if self.prev_time is None or timestamp is None:
            self.prev_time = timestamp if timestamp else time.time()
            return self.x_filt.filter(val)
        
        dt = timestamp - self.prev_time
        if dt <= 0:
            return self.x_filt.prev_val
        
        # Estimate derivative
        prev_x = self.x_filt.prev_val
        dx = (val - prev_x) / dt
        edx = self.dx_filt.filter(dx)
        
        # Calculate adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, dt)
        
        self.prev_time = timestamp
        return self.x_filt.filter(val) # LowPassFilter needs to use the dynamic alpha

    # Override for dynamic alpha in the internal filters
    def update(self, val: float, timestamp: float) -> float:
        if self.prev_time is None:
            self.prev_time = timestamp
            self.x_filt.prev_val = val
            self.dx_filt.prev_val = 0.0
            return val
        
        dt = timestamp - self.prev_time
        if dt <= 0: return self.x_filt.prev_val
        
        # 1. Filter derivative
        dx = (val - self.x_filt.prev_val) / dt
        d_alpha = self._alpha(self.d_cutoff, dt)
        edx = d_alpha * dx + (1.0 - d_alpha) * self.dx_filt.prev_val
        self.dx_filt.prev_val = edx
        
        # 2. Filter value with adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, dt)
        result = alpha * val + (1.0 - alpha) * self.x_filt.prev_val
        
        self.x_filt.prev_val = result
        self.prev_time = timestamp
        
        return result
