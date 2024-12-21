import numpy as np
from raytracer.ray import Ray

class Camera:
    def __init__(self, position, look_at, up, fov, aspect_ratio):
        self.position = np.array(position)
        self.forward = self._normalize(np.array(look_at) - self.position)
        self.right = self._normalize(np.cross(self.forward, up))
        self.up = np.cross(self.right, self.forward)
        self.fov = np.radians(fov)
        self.aspect_ratio = aspect_ratio
        self.half_width = np.tan(self.fov / 2)
        self.half_height = self.half_width / self.aspect_ratio

    def _normalize(self, v):
        return v / np.linalg.norm(v)

    def get_ray(self, u, v):
        direction = self.forward + u * self.half_width * self.right + v * self.half_height * self.up
        direction = self._normalize(direction)
        return Ray(self.position, direction)
