from numba import cuda, float32  # Correctly import float32 for usage
from math import sqrt  # Import sqrt for compatibility in both host and device
import numpy as np
from raytracer.geometry import Sphere, Plane  # Import the geometry classes

@cuda.jit
def render_kernel(scene_data, camera_data, output, width, height, samples_per_pixel, light_data):
    x, y = cuda.grid(2)

    if x >= width or y >= height:
        return

    # Compute normalized screen coordinates
    u = (x + 0.5) / width
    v = (y + 0.5) / height

    # Extract camera data
    origin = cuda.local.array(3, dtype=float32)
    for i in range(3):
        origin[i] = camera_data[i]

    direction = cuda.local.array(3, dtype=float32)
    for i in range(3):
        direction[i] = (
            camera_data[3 + i]
            + (2 * u - 1) * camera_data[6 + i]
            + (2 * v - 1) * camera_data[9 + i]
        )

    # Normalize direction
    magnitude = 0.0
    for i in range(3):
        magnitude += direction[i] * direction[i]
    magnitude = sqrt(magnitude)
    for i in range(3):
        direction[i] /= magnitude

    # Extract light data
    light_pos = cuda.local.array(3, dtype=float32)
    light_color = cuda.local.array(3, dtype=float32)
    light_intensity = light_data[6]
    for i in range(3):
        light_pos[i] = light_data[i]
        light_color[i] = light_data[3 + i]

    # Initialize hit color and distance
    color = cuda.local.array(3, dtype=float32)
    for i in range(3):
        color[i] = 0.0  # Initialize to black

    ambient_light = cuda.local.array(3, dtype=float32)
    for i in range(3):
        ambient_light[i] = 0.2  # Slightly reduced ambient light strength

    for i in range(3):
        color[i] += ambient_light[i]  # Add ambient lighting

    closest_t = float('inf')

    # Reflection parameters
    max_reflections = 1
    reflection_strength = 0.4  # Reduce reflection intensity

    # Intersection tests
    for reflection_count in range(max_reflections + 1):
        hit_object = -1
        hit_normal = cuda.local.array(3, dtype=float32)
        hit_point = cuda.local.array(3, dtype=float32)
        for obj_idx, obj in enumerate(scene_data):
            if obj[0] == 1:  # Sphere type
                center = obj[1:4]
                radius = obj[4]
                diff = cuda.local.array(3, dtype=float32)
                for i in range(3):
                    diff[i] = origin[i] - center[i]
                b = 2.0 * (direction[0] * diff[0] + direction[1] * diff[1] + direction[2] * diff[2])
                c = diff[0]**2 + diff[1]**2 + diff[2]**2 - radius**2
                discriminant = b * b - 4 * c
                if discriminant >= 0.0:
                    t = (-b - sqrt(discriminant)) / 2.0
                    if 0 < t < closest_t:
                        closest_t = t
                        hit_object = obj_idx
                        for i in range(3):
                            hit_point[i] = origin[i] + t * direction[i]
                            hit_normal[i] = (hit_point[i] - center[i]) / radius
                        # Normalize the normal vector
                        normal_magnitude = sqrt(hit_normal[0]**2 + hit_normal[1]**2 + hit_normal[2]**2)
                        for i in range(3):
                            hit_normal[i] /= normal_magnitude

            elif obj[0] == 2:  # Plane type
                point = obj[1:4]
                normal = obj[4:7]
                denom = (direction[0] * normal[0] +
                         direction[1] * normal[1] +
                         direction[2] * normal[2])
                if abs(denom) > 1e-6:  # Avoid division by zero
                    diff = cuda.local.array(3, dtype=float32)
                    for i in range(3):
                        diff[i] = point[i] - origin[i]
                    t = (diff[0] * normal[0] +
                         diff[1] * normal[1] +
                         diff[2] * normal[2]) / denom
                    if 0 < t < closest_t:
                        closest_t = t
                        hit_object = obj_idx
                        for i in range(3):
                            hit_point[i] = origin[i] + t * direction[i]
                            hit_normal[i] = normal[i]

        if hit_object == -1:  # No intersection
            break

        # Calculate lighting for the hit object
        light_dir = cuda.local.array(3, dtype=float32)
        for i in range(3):
            light_dir[i] = light_pos[i] - hit_point[i]
        light_mag = 0.0
        for i in range(3):
            light_mag += light_dir[i] * light_dir[i]
        light_mag = sqrt(light_mag)
        for i in range(3):
            light_dir[i] /= light_mag
        # Apply light attenuation
        attenuation = light_intensity / (4 * 3.14159 * max(light_mag * light_mag, 1e-3))
        shadow_ray_origin = cuda.local.array(3, dtype=float32)
        for i in range(3):
            shadow_ray_origin[i] = hit_point[i] + 1e-3 * hit_normal[i]
        in_shadow = False
        for shadow_obj in scene_data:
            if shadow_obj[0] == 1:  # Sphere
                shadow_center = shadow_obj[1:4]
                shadow_radius = shadow_obj[4]
                shadow_diff = cuda.local.array(3, dtype=float32)
                for i in range(3):
                    shadow_diff[i] = shadow_ray_origin[i] - shadow_center[i]
                b = 2.0 * (light_dir[0] * shadow_diff[0] + light_dir[1] * shadow_diff[1] + light_dir[2] * shadow_diff[2])
                c = shadow_diff[0]**2 + shadow_diff[1]**2 + shadow_diff[2]**2 - shadow_radius**2
                shadow_discriminant = b * b - 4 * c
                if shadow_discriminant >= 0.0:
                    in_shadow = True
                    break

        if not in_shadow:
            dot = max(0.0, light_dir[0] * hit_normal[0] + light_dir[1] * hit_normal[1] + light_dir[2] * hit_normal[2])

            # Specular lighting
            view_dir = cuda.local.array(3, dtype=float32)
            reflect_dir = cuda.local.array(3, dtype=float32)
            for i in range(3):
                view_dir[i] = -direction[i]
                reflect_dir[i] = 2 * dot * hit_normal[i] - light_dir[i]
            spec = pow(max(0.0, reflect_dir[0] * view_dir[0] + reflect_dir[1] * view_dir[1] + reflect_dir[2] * view_dir[2]), 32)

            obj_color = scene_data[hit_object][5:8]
            for i in range(3):
                color[i] += obj_color[i] * light_color[i] * attenuation * dot
                color[i] += spec * light_color[i] * 0.5  # Add specular highlight

            # Add texture pattern to walls
            if obj[0] == 2:  # Plane type
                texture_u = (hit_point[0] % 1.0)  # Create stripes based on position
                texture_v = (hit_point[2] % 1.0)
                texture_intensity = 0.5 * (1 + (texture_u * texture_v))  # Modulate intensity
                for i in range(3):
                    color[i] *= texture_intensity  # Apply texture pattern

        # Reflection
        if reflection_count < max_reflections:
            reflect_dir = cuda.local.array(3, dtype=float32)
            for i in range(3):
                reflect_dir[i] = direction[i] - 2.0 * (direction[0] * hit_normal[0] +
                                                       direction[1] * hit_normal[1] +
                                                       direction[2] * hit_normal[2]) * hit_normal[i]
            for i in range(3):
                origin[i] = hit_point[i]
                direction[i] = reflect_dir[i]
            closest_t = float('inf')
            reflected_color = cuda.local.array(3, dtype=float32)
            for i in range(3):
                reflected_color[i] = color[i] * reflection_strength
            for i in range(3):
                color[i] = min(color[i] + reflected_color[i], 1.0)
        else:
            break

    # Write the color to the output
    output[y, x, 0] = min(color[0], 1.0)
    output[y, x, 1] = min(color[1], 1.0)
    output[y, x, 2] = min(color[2], 1.0)


class Renderer:
    def __init__(self, width, height, samples_per_pixel=1):
        self.width = width
        self.height = height
        self.samples_per_pixel = samples_per_pixel

    def render(self, scene, camera, light_pos, light_color, light_intensity):
        # Allocate output array
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
                scene_data_np.append([
                    1,                     # Type identifier for Sphere
                    *obj.center,           # Center (3 values)
                    obj.radius,            # Radius
                    0.0, 0.0, 0.0,         # Padding for unused fields
                    *obj.material.color    # Material color (3 values)
                ])
            elif isinstance(obj, Plane):
                scene_data_np.append([
                    2,                     # Type identifier for Plane
                    *obj.point,            # Point on the plane (3 values)
                    *obj.normal,           # Normal (3 values)
                    0.0,                   # Padding for unused field
                    *obj.material.color    # Material color (3 values)
                ])
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        # Convert to NumPy array
        scene_data_np = np.array(scene_data_np, dtype=np.float32)
        d_scene = cuda.to_device(scene_data_np)

        # Light data
        light_data = np.array(light_pos + light_color + [light_intensity], dtype=np.float32)
        d_light = cuda.to_device(light_data)

        # Define thread and grid sizes
        threads_per_block = (16, 16)
        blocks_per_grid_x = (self.width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self.height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch kernel
        render_kernel[blocks_per_grid, threads_per_block](
            d_scene, d_camera, d_output, self.width, self.height, self.samples_per_pixel,
            d_light
        )

        # Copy output back to host
        output = d_output.copy_to_host()
        return output
