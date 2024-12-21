import numpy as np

class Material:
    def __init__(self, color, reflectivity=0, texture=None):
        self.color = np.array(color)  # Ensure the color is a NumPy array
        self.reflectivity = reflectivity
        self.texture = texture

    def get_color(self, point):
        if self.texture:
            return np.array(self.texture(point))  # Ensure the texture result is a NumPy array
        return self.color

    def checkerboard_texture(point):
        scale = 2.0
        if (int(point[0] * scale) + int(point[2] * scale)) % 2 == 0:
            return np.array([1, 1, 1])  # White
        else:
            return np.array([0, 0, 0])  # Black

