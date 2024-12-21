import numpy as np

class BoundingBox:
    def __init__(self, min_point, max_point):
        self.min_point = np.array(min_point)
        self.max_point = np.array(max_point)

    def intersect(self, ray):
        """
        Checks if a ray intersects the bounding box.
        """
        inv_dir = 1.0 / ray.direction
        t_min = (self.min_point - ray.origin) * inv_dir
        t_max = (self.max_point - ray.origin) * inv_dir

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = np.max(t1)
        t_far = np.min(t2)

        return t_near < t_far and t_far > 0

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """
        Calculate the intersection of a ray with the sphere.
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
        return (point - self.center) / np.linalg.norm(point - self.center)

    def bounding_box(self):
        min_point = self.center - self.radius
        max_point = self.center + self.radius
        return BoundingBox(min_point, max_point)

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
        return self.normal

    def bounding_box(self):
        size = 1e6  # Large value to approximate infinite plane
        min_point = np.array([-size, -size, -size])
        max_point = np.array([size, size, size])
        return BoundingBox(min_point, max_point)

class BVHNode:
    def __init__(self, objects):
        if len(objects) == 1:
            self.objects = objects
            self.bbox = objects[0].bounding_box()
        else:
            # Split objects and create child nodes
            objects.sort(key=lambda obj: obj.bounding_box().min_point[0])
            mid = len(objects) // 2
            self.left = BVHNode(objects[:mid])
            self.right = BVHNode(objects[mid:])
            self.bbox = BoundingBox(
                np.minimum(self.left.bbox.min_point, self.right.bbox.min_point),
                np.maximum(self.left.bbox.max_point, self.right.bbox.max_point),
            )

    def intersect(self, ray):
        if not self.bbox.intersect(ray):
            return None, float('inf')
        if hasattr(self, 'objects'):
            closest_object, closest_t = None, float('inf')
            for obj in self.objects:
                t = obj.intersect(ray)
                if t and t < closest_t:
                    closest_object, closest_t = obj, t
            return closest_object, closest_t
        else:
            left_hit, left_t = self.left.intersect(ray)
            right_hit, right_t = self.right.intersect(ray)
            if left_t < right_t:
                return left_hit, left_t
            return right_hit, right_t

