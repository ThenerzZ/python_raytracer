import numpy as np
from raytracer.ray import Ray  # Import the Ray class

class Camera:
    def __init__(self, position, look_at, up, fov, aspect_ratio):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up = np.array(up)
        self.fov = np.radians(fov)  # Convert FOV to radians
        self.aspect_ratio = aspect_ratio

        # Compute camera basis vectors
        self.forward = self._normalize(self.look_at - self.position)
        self.right = self._normalize(np.cross(self.forward, self.up))
        self.true_up = np.cross(self.right, self.forward)

        # Precompute viewport dimensions
        self.viewport_height = 2.0 * np.tan(self.fov / 2)
        self.viewport_width = self.viewport_height * self.aspect_ratio

    def _normalize(self, v):
        return v / np.linalg.norm(v)

    def get_ray(self, u, v):
        direction = (
            self.forward +
            u * self.viewport_width * self.right +
            v * self.viewport_height * self.true_up
        )
        return Ray(self.position, self._normalize(direction))  # Return a Ray object
