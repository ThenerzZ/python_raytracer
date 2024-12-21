import numpy as np

def checkerboard_texture(point):
    """
    Generate a checkerboard pattern based on the given point.
    """
    scale = 2.0  # Adjust for larger/smaller squares
    if (int(point[0] * scale) + int(point[2] * scale)) % 2 == 0:
        return np.array([1, 1, 1])  # White
    else:
        return np.array([0, 0, 0])  # Black
