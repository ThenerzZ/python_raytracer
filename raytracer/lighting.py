import numpy as np

class Light:
    def __init__(self, position, intensity):
        self.position = np.array(position)
        self.intensity = intensity
