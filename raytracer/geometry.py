### Updated `geometry.py`
import numpy as np
from raytracer.materials import Material

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        if t1 > 0:
            return t1
        if t2 > 0:
            return t2
        return None

    def get_normal(self, point):
        return (point - self.center) / np.linalg.norm(point - self.center)


class Plane:
    def __init__(self, point, normal, material):
        self.point = np.array(point)
        self.normal = self._normalize(normal)
        self.material = material

    def _normalize(self, v):
        return v / np.linalg.norm(v)

    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) > 1e-6:
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t > 0:
                return t
        return None

    def get_normal(self, _):
        return self.normal


class Box:
    def __init__(self, min_point, max_point, material):
        self.min_point = np.array(min_point)
        self.max_point = np.array(max_point)
        self.material = material

    def intersect(self, ray):
        inv_dir = 1.0 / ray.direction
        t_min = (self.min_point - ray.origin) * inv_dir
        t_max = (self.max_point - ray.origin) * inv_dir

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near > t_far or t_far < 0:
            return None
        return t_near

    def get_normal(self, point):
        epsilon = 1e-4
        if abs(point[0] - self.min_point[0]) < epsilon:
            return np.array([-1, 0, 0])
        if abs(point[0] - self.max_point[0]) < epsilon:
            return np.array([1, 0, 0])
        if abs(point[1] - self.min_point[1]) < epsilon:
            return np.array([0, -1, 0])
        if abs(point[1] - self.max_point[1]) < epsilon:
            return np.array([0, 1, 0])
        if abs(point[2] - self.min_point[2]) < epsilon:
            return np.array([0, 0, -1])
        if abs(point[2] - self.max_point[2]) < epsilon:
            return np.array([0, 0, 1])
        return np.array([0, 0, 0])  # Fallback