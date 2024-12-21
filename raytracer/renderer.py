from concurrent.futures import ProcessPoolExecutor
import numpy as np
from raytracer.ray import Ray

def render_section(args):
    start_y, end_y, width, height, samples_per_pixel, aspect_ratio, camera, scene, renderer = args
    section_pixels = np.zeros((end_y - start_y, width, 3))
    for y in range(start_y, end_y):
        for x in range(width):
            color = np.zeros(3)
            for _ in range(samples_per_pixel):
                u = ((x + np.random.random()) / width - 0.5) * 2 * aspect_ratio
                v = (0.5 - (y + np.random.random()) / height) * 2
                ray = camera.get_ray(u, v)
                color += renderer.trace_ray(ray, scene)
            section_pixels[y - start_y, x] = np.clip(color / samples_per_pixel, 0, 1)
    return section_pixels

class Renderer:
    def __init__(self, width, height, max_depth=5, samples_per_pixel=1):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.samples_per_pixel = samples_per_pixel

    def render(self, scene, camera):
        aspect_ratio = self.width / self.height
        pixels = np.zeros((self.height, self.width, 3))

        # Divide the image into sections
        num_processes = 4
        section_height = self.height // num_processes
        sections = [(i * section_height, (i + 1) * section_height) for i in range(num_processes)]
        if self.height % num_processes != 0:
            sections[-1] = (sections[-1][0], self.height)

        # Prepare arguments for multiprocessing
        args = [
            (start, end, self.width, self.height, self.samples_per_pixel, aspect_ratio, camera, scene, self)
            for start, end in sections
        ]

        # Render sections in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(render_section, args))

        # Combine results
        for i, section in enumerate(results):
            start_y, end_y = sections[i]
            pixels[start_y:end_y, :, :] = section

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

            # Add reflections
            if material.reflectivity > 0:
                reflection_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
                reflection_ray = Ray(hit_point + 1e-4 * reflection_dir, reflection_dir)
                reflection_color = self.trace_ray(reflection_ray, scene, depth + 1)
                color = (1 - material.reflectivity) * color + material.reflectivity * reflection_color

            return color

        return np.array([0, 0, 0])  # Background color

    def compute_lighting(self, point, normal, scene, view_dir):
        intensity = 0
        for light in scene.lights:
            light_dir = light.position - point
            light_dist = np.linalg.norm(light_dir)
            light_dir = light_dir / light_dist

            # Shadow check
            shadow_ray = Ray(point + 1e-4 * light_dir, light_dir)
            shadow_hit, shadow_t = scene.intersect(shadow_ray)
            if shadow_hit and shadow_t < light_dist:
                continue

            # Diffuse
            intensity += max(0, np.dot(normal, light_dir)) * light.intensity

            # Specular
            reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
            intensity += max(0, np.dot(view_dir, reflect_dir)) ** 50 * light.intensity  # Shininess factor

        return intensity
