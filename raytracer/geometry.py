import numpy as np

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """
        Compute the intersection of the ray with the sphere.
        """
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
        """
        Calculate the normal vector at a given point on the sphere's surface.
        """
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
        if abs(denom) > 1e-6:  # Avoid division by zero
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t > 0:
                return t
        return None

    def get_normal(self, _):
        """
        Return the normal of the plane. For a plane, the normal is constant
        and does not depend on the hit point.
        """
        return self.normal
