import numpy as np

def approximate_first_crossing(time_array: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    """Approximates the first time the interpolated values cross above the given threshold using linear interpolation."""
    for i in range(1, len(values)):
        if values[i - 1] < threshold and values[i] >= threshold:
            # Performs linear interpolation between points (x0, y0) and (x1, y1)
            x0, x1 = time_array[i - 1], time_array[i]
            y0, y1 = values[i - 1], values[i]
            # Estimates crossing time
            t_cross = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            return float(t_cross)
    return None