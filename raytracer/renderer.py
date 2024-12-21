import numpy as np
from raytracer.ray import Ray


class Renderer:
    def __init__(self, width, height, max_depth=5):
        self.width = width
        self.height = height
        self.max_depth = max_depth

    def render(self, scene, camera):
        aspect_ratio = self.width / self.height
        pixels = np.zeros((self.height, self.width, 3))

        for y in range(self.height):
            for x in range(self.width):
                # Map pixel coordinates to [-1, 1] in both dimensions
                u = (x / self.width - 0.5) * 2 * aspect_ratio
                v = (0.5 - y / self.height) * 2
                ray = camera.get_ray(u, v)
                color = self.trace_ray(ray, scene)
                pixels[y, x] = np.clip(color, 0, 1)

        return pixels

    def trace_ray(self, ray, scene, depth=0):
        if depth > self.max_depth:
            return np.array([0, 0, 0])  # Background color

        hit_object, t = scene.intersect(ray)
        if hit_object:
            hit_point = ray.point_at_parameter(t)
            normal = hit_object.get_normal(hit_point)
            view_dir = -ray.direction

            # Compute lighting
            material = hit_object.material
            color = material.get_color(hit_point) * self.compute_lighting(hit_point, normal, scene, view_dir)

            # Add reflections if applicable
            if material.reflectivity > 0:
                reflection_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
                reflection_ray = Ray(hit_point + 1e-4 * reflection_dir, reflection_dir)
                reflection_color = self.trace_ray(reflection_ray, scene, depth + 1)
                color = (1 - material.reflectivity) * color + material.reflectivity * reflection_color

            return color

        return np.array([0, 0, 0])  # Background color

    def compute_lighting(self, point, normal, scene, view_dir):
        """
        Calculate lighting at a given point using diffuse and specular components.
        """
        intensity = 0
        for light in scene.lights:
            # Direction to light
            light_dir = light.position - point
            light_dist = np.linalg.norm(light_dir)
            light_dir = light_dir / light_dist

            # Shadow check
            shadow_ray = Ray(point + 1e-4 * light_dir, light_dir)
            shadow_hit, shadow_t = scene.intersect(shadow_ray)
            if shadow_hit and shadow_t < light_dist:
                continue  # In shadow

            # Diffuse lighting (Lambertian)
            diffuse_intensity = max(np.dot(normal, light_dir), 0)
            intensity += light.intensity * diffuse_intensity

            # Specular lighting (Phong reflection model)
            reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
            specular_intensity = max(np.dot(view_dir, reflect_dir), 0) ** 50  # 50: shininess factor
            intensity += light.intensity * specular_intensity

        return intensity
