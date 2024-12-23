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
    def __init__(self, min_corner, max_corner, material):
        """
        A 3D box defined by its minimum and maximum corners.
        :param min_corner: Minimum corner of the box [x_min, y_min, z_min]
        :param max_corner: Maximum corner of the box [x_max, y_max, z_max]
        :param material: Material of the box
        """
        self.min_corner = np.array(min_corner, dtype=np.float32)
        self.max_corner = np.array(max_corner, dtype=np.float32)
        self.material = material

    def intersect(self, ray_origin, ray_direction):
        """
        Ray-box intersection using the slab method.
        :param ray_origin: Origin of the ray
        :param ray_direction: Direction of the ray
        :return: (t_min, t_max) or None if no intersection
        """
        t_min = (self.min_corner - ray_origin) / ray_direction
        t_max = (self.max_corner - ray_origin) / ray_direction

        t_min = np.minimum(t_min, t_max)
        t_max = np.maximum(t_min, t_max)

        t0 = np.max(t_min)
        t1 = np.min(t_max)

        if t0 > t1 or t1 < 0:
            return None
        return t0, t1
