import numpy as np

class Box:
    def __init__(self, min_corner, max_corner, material):
        self.min_corner = np.array(min_corner, dtype=np.float32)
        self.max_corner = np.array(max_corner, dtype=np.float32)
        self.material = material

    def serialize(self):
        """
        Convert the Box object to a serializable format for the renderer.
        Format:
        [type, min_x, min_y, min_z, max_x, max_y, max_z, material_color_r, material_color_g, material_color_b, shininess, reflectivity, transparency, refractive_index]
        """
        return [
            3,  # Type identifier for Box
            *self.min_corner,
            *self.max_corner,
            *self.material.color,
            self.material.shininess,
            self.material.reflectivity,
            self.material.transparency,
            self.material.refractive_index
        ]

    def intersect(self, ray_origin, ray_direction):
        """
        Intersection for an axis-aligned bounding box (AABB) using the 'slab' method.

        Returns the closest positive t-value of intersection, or None if no intersection.
        """
        t_min = -np.inf
        t_max = np.inf

        origin = np.array(ray_origin, dtype=np.float32)
        direction = np.array(ray_direction, dtype=np.float32)

        for i in range(3):
            if abs(direction[i]) < 1e-7:
                if origin[i] < self.min_corner[i] or origin[i] > self.max_corner[i]:
                    return None
            else:
                t1 = (self.min_corner[i] - origin[i]) / direction[i]
                t2 = (self.max_corner[i] - origin[i]) / direction[i]
                t_near, t_far = min(t1, t2), max(t1, t2)

                t_min = max(t_min, t_near)
                t_max = min(t_max, t_far)

                if t_min > t_max:
                    return None

        if t_min < 0:
            if t_max < 0:
                return None
            return t_max
        return t_min


class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.material = material

    def serialize(self):
        """
        Convert the Sphere object to a serializable format for the renderer.
        Format:
        [type, center_x, center_y, center_z, radius, material_color_r, material_color_g, material_color_b, shininess, reflectivity]
        """
        return [
            1,  # Type identifier for Sphere
            *self.center,
            self.radius,
            *self.material.color,
            self.material.shininess,
            self.material.reflectivity,
        ]

    def intersect(self, ray_origin, ray_direction):
        """
        Ray-sphere intersection using the standard quadratic formula.

        Returns the closest positive t-value of intersection, or None if no intersection.
        """
        origin = np.array(ray_origin, dtype=np.float32)
        direction = np.array(ray_direction, dtype=np.float32)

        L = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(L, direction)
        c = np.dot(L, L) - (self.radius ** 2)

        discriminant = b**2 - 4.0 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        t1, t2 = min(t1, t2), max(t1, t2)

        if t2 < 0:
            return None
        if t1 < 0:
            return t2
        return t1
