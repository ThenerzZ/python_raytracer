import numpy as np
from raytracer.ray import Ray

class Camera:
    def __init__(self, position, look_at, up, fov, aspect_ratio):
        self.position = np.array(position, dtype=np.float32)
        self.forward = np.array(look_at, dtype=np.float32) - self.position
        self.forward /= np.linalg.norm(self.forward)

        self.right = np.cross(self.forward, np.array(up, dtype=np.float32))
        self.right /= np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.forward)

        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.half_height = np.tan(np.radians(self.fov) / 2)
        self.half_width = self.aspect_ratio * self.half_height

    def generate_ray(self, u, v):
        """Generate a ray from the camera through the screen at normalized coords (u, v)."""
        direction = (
            self.forward
            + (2 * u - 1) * self.half_width * self.right
            + (2 * v - 1) * self.half_height * self.up
        )
        direction /= np.linalg.norm(direction)
        return self.position, direction
