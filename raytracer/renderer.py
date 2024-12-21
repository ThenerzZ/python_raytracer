import numpy as np
from raytracer.ray import Ray

class Renderer:
    def __init__(self, width, height, max_depth=5, samples_per_pixel=1, mode='preview'):
        self.base_width = width
        self.base_height = height
        self.max_depth = max_depth
        self.samples_per_pixel = samples_per_pixel

        if mode == 'preview':
            self.width = self.base_width // 2
            self.height = self.base_height // 2
            self.samples_per_pixel = min(4, self.samples_per_pixel)
        elif mode == 'high_quality':
            self.width = self.base_width
            self.height = self.base_height
            self.samples_per_pixel = max(64, self.samples_per_pixel)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def render(self, scene, camera):
        aspect_ratio = self.width / self.height
        pixels = np.zeros((self.height, self.width, 3))

        for y in range(self.height):
            for x in range(self.width):
                color = np.zeros(3)
                for _ in range(self.samples_per_pixel):
                    u = ((x + np.random.random()) / self.width - 0.5) * 2 * aspect_ratio
                    v = (0.5 - (y + np.random.random()) / self.height) * 2
                    ray = camera.get_ray(u, v)
                    color += self.trace_ray(ray, scene)

                pixels[y, x] = np.clip(color / self.samples_per_pixel, 0, 1)

        # Apply gamma correction
        gamma = 2.2
        pixels = np.power(pixels, 1 / gamma)

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
            base_color = material.get_color(hit_point)
            lighting = self.compute_lighting(hit_point, normal, scene, view_dir)
            color = base_color * lighting

            # Add reflections
            if material.reflectivity > 0:
                reflection_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
                reflection_ray = Ray(hit_point + 1e-4 * reflection_dir, reflection_dir)
                reflection_color = self.trace_ray(reflection_ray, scene, depth + 1)
                color += material.reflectivity * reflection_color

            # Add transparency/refraction
            if material.transparency > 0:
                n = material.refractive_index
                if np.dot(view_dir, normal) > 0:  # Inside the object
                    n = 1 / n
                    normal = -normal
                cos_i = -np.dot(normal, view_dir)
                sin_t2 = (n ** 2) * (1 - cos_i ** 2)
                if sin_t2 <= 1:  # Total internal reflection check
                    cos_t = np.sqrt(1 - sin_t2)
                    refraction_dir = n * view_dir + (n * cos_i - cos_t) * normal
                    refraction_ray = Ray(hit_point + 1e-4 * refraction_dir, refraction_dir)
                    refraction_color = self.trace_ray(refraction_ray, scene, depth + 1)
                    color += material.transparency * refraction_color

            return np.clip(color, 0, 1)

        return np.array([0, 0, 0])  # Background color

    def compute_lighting(self, point, normal, scene, view_dir):
        intensity = 0
        shadow_samples = 16  # For soft shadows

        for light in scene.lights:
            light_dir = light.position - point
            light_dist = np.linalg.norm(light_dir)
            light_dir = light_dir / light_dist

            shadow_intensity = 0
            for _ in range(shadow_samples):
                jitter = np.random.normal(scale=0.1, size=3)  # Jitter for soft shadows
                shadow_ray_dir = light_dir + jitter
                shadow_ray_dir = shadow_ray_dir / np.linalg.norm(shadow_ray_dir)

                shadow_ray = Ray(point + 1e-4 * shadow_ray_dir, shadow_ray_dir)
                shadow_hit, shadow_t = scene.intersect(shadow_ray)
                if not shadow_hit or shadow_t > light_dist:
                    shadow_intensity += 1

            shadow_intensity /= shadow_samples

            # Diffuse lighting
            diffuse = max(np.dot(normal, light_dir), 0) * shadow_intensity
            intensity += light.intensity * diffuse

            # Specular lighting
            reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
            specular = max(np.dot(view_dir, reflect_dir), 0) ** 50  # Shininess factor
            intensity += light.intensity * specular

        # Ambient occlusion
        ao = self.compute_ambient_occlusion(point, normal, scene)
        intensity *= ao

        return intensity

    def compute_ambient_occlusion(self, point, normal, scene, samples=8, radius=1.0):
        occlusion = 0
        for _ in range(samples):
            sample_dir = np.random.normal(size=3)
            sample_dir = sample_dir / np.linalg.norm(sample_dir)
            if np.dot(sample_dir, normal) < 0:
                sample_dir = -sample_dir  # Ensure hemisphere sampling

            sample_point = point + radius * sample_dir
            occlusion_ray = Ray(point + 1e-4 * sample_dir, sample_dir)
            hit, _ = scene.intersect(occlusion_ray)
            if hit:
                occlusion += 1

        return 1 - (occlusion / samples)
