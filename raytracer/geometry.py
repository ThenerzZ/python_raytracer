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
        [type, min_x, min_y, min_z, max_x, max_y, max_z, material_color_r, material_color_g, material_color_b, shininess, reflectivity]
        """
        return [
            2,  # Type identifier for Box
            *self.min_corner,
            *self.max_corner,
            *self.material.color,
            self.material.shininess,
            self.material.reflectivity,
        ]

    def intersect(self, ray_origin, ray_direction):
        """
        Intersection logic for the Box (optional, based on current requirements).
        Implement axis-aligned bounding box (AABB) intersection here.
        """
        pass

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
        Intersection logic for the Sphere (optional, based on current requirements).
        Implement sphere-ray intersection here.
        """
        pass
