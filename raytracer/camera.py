import numpy as np

class Camera:
    def __init__(self, position, look_at, up, fov, aspect_ratio):
        self.position = np.array(position, dtype=np.float32)
        self.forward = np.array(look_at, dtype=np.float32) - self.position
        self.forward = self.forward / np.linalg.norm(self.forward)

        self.right = np.cross(self.forward, np.array(up, dtype=np.float32))
        self.right = self.right / np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.forward)
        self.fov = fov
        self.aspect_ratio = aspect_ratio

    def generate_ray(self, u, v):
        alpha = np.tan(np.radians(self.fov) / 2.0) * (2 * u - 1) * self.aspect_ratio
        beta = np.tan(np.radians(self.fov) / 2.0) * (1 - 2 * v)
        direction = self.forward + alpha * self.right + beta * self.up
        direction = direction / np.linalg.norm(direction)
        return Ray(self.position, direction)
