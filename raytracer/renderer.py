import numpy as np
from numba import cuda, float32
from math import sqrt
from raytracer.geometry import Sphere, Plane
from raytracer.materials import Material

import numba
from numba import cuda
import numpy as np

@cuda.jit
def render_kernel(scene_data, camera_data, output, width, height, samples_per_pixel, light_data):
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    # Initialize color
    color = cuda.local.array(3, dtype=float32)
    for i in range(3):
        color[i] = 0.0

    # Add ambient light
    ambient_light = cuda.local.array(3, dtype=float32)
    ambient_light[0] = 0.2  # Ambient intensity
    ambient_light[1] = 0.2
    ambient_light[2] = 0.2
    for i in range(3):
        color[i] += ambient_light[i]

    # Ray initialization
    ray_origin = cuda.local.array(3, dtype=float32)
    ray_direction = cuda.local.array(3, dtype=float32)
    for i in range(3):
        ray_origin[i] = camera_data[i]
        ray_direction[i] = (
            camera_data[3 + i]
            + (2 * (x / width) - 1) * camera_data[6 + i]
            + (2 * (y / height) - 1) * camera_data[9 + i]
        )
    # Normalize ray direction
    magnitude = 0.0
    for i in range(3):
        magnitude += ray_direction[i] * ray_direction[i]
    magnitude = sqrt(magnitude)
    for i in range(3):
        ray_direction[i] /= magnitude

    # Ray-object intersections
    closest_t = float('inf')
    hit_object = -1
    hit_point = cuda.local.array(3, dtype=float32)
    hit_normal = cuda.local.array(3, dtype=float32)

    for obj_idx in range(scene_data.shape[0]):
        obj = scene_data[obj_idx]
        if obj[0] == 1:  # Sphere
            sphere_center = cuda.local.array(3, dtype=float32)
            for i in range(3):
                sphere_center[i] = obj[1 + i]
            sphere_radius = obj[4]

            oc = cuda.local.array(3, dtype=float32)
            for i in range(3):
                oc[i] = ray_origin[i] - sphere_center[i]

            a, b, c = 0.0, 0.0, 0.0
            for i in range(3):
                a += ray_direction[i] * ray_direction[i]
                b += 2.0 * oc[i] * ray_direction[i]
                c += oc[i] * oc[i]
            c -= sphere_radius * sphere_radius

            discriminant = b * b - 4 * a * c
            if discriminant > 0:
                t = (-b - sqrt(discriminant)) / (2.0 * a)
                if t > 0 and t < closest_t:
                    closest_t = t
                    hit_object = obj_idx
                    for i in range(3):
                        hit_point[i] = ray_origin[i] + t * ray_direction[i]
                        hit_normal[i] = (hit_point[i] - sphere_center[i]) / sphere_radius

    # Shading calculations
    if hit_object != -1:
        light_dir = cuda.local.array(3, dtype=float32)
        for i in range(3):
            light_dir[i] = light_data[i] - hit_point[i]
        light_mag = 0.0
        for i in range(3):
            light_mag += light_dir[i] * light_dir[i]
        light_mag = sqrt(light_mag)
        for i in range(3):
            light_dir[i] /= light_mag

        # Diffuse shading
        dot = max(0.0, sum(hit_normal[i] * light_dir[i] for i in range(3)))
        for i in range(3):
            color[i] += dot * light_data[6] * light_data[3 + i]

        # Specular shading
        view_dir = cuda.local.array(3, dtype=float32)
        halfway_dir = cuda.local.array(3, dtype=float32)
        for i in range(3):
            view_dir[i] = -ray_direction[i]
            halfway_dir[i] = view_dir[i] + light_dir[i]
        halfway_mag = sqrt(sum(halfway_dir[i] * halfway_dir[i] for i in range(3)))
        for i in range(3):
            halfway_dir[i] /= halfway_mag
        spec_dot = max(0.0, sum(hit_normal[i] * halfway_dir[i] for i in range(3)))
        specular = pow(spec_dot, 32)  # Adjust shininess if needed
        for i in range(3):
            color[i] += specular * light_data[6] * light_data[3 + i]

    # Write to output
    for i in range(3):
        output[y, x, i] = min(1.0, max(0.0, color[i]))  # Clamp color

class Renderer:
    def __init__(self, width, height, samples_per_pixel=1):
        self.width = width
        self.height = height
        self.samples_per_pixel = samples_per_pixel

    def render(self, scene, camera):
        output = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Flatten output for GPU compatibility
        d_output = cuda.to_device(output)

        # Camera data
        camera_data = np.hstack(
            [camera.position, camera.forward, camera.right, camera.up]
        ).astype(np.float32)
        d_camera = cuda.to_device(camera_data)

        # Serialize scene objects
        scene_data_np = []
        for obj in scene.get_objects():
            if isinstance(obj, Sphere):
                scene_data_np.append([1, *obj.center, obj.radius, 0, 0, 0, *obj.material.color])
            elif isinstance(obj, Plane):
                scene_data_np.append([2, *obj.point, *obj.normal, 0, *obj.material.color])
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        scene_data_np = np.array(scene_data_np, dtype=np.float32)
        d_scene = cuda.to_device(scene_data_np)

        # Serialize lights
        light_data = []
        for light in scene.get_lights():
            light_data.extend(light["position"] + light["color"] + [light["intensity"]])

        if not light_data:
            raise ValueError("No lights found in the scene!")

        d_lights = cuda.to_device(np.array(light_data, dtype=np.float32))

        # Define thread and grid sizes
        threads_per_block = (16, 16)
        blocks_per_grid_x = (self.width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self.height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch kernel
        render_kernel[blocks_per_grid, threads_per_block](
            d_scene, d_camera, d_output, self.width, self.height, self.samples_per_pixel, d_lights
        )

        # Copy output back to host
        output = d_output.copy_to_host()
        return output
