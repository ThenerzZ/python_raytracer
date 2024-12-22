from numba import cuda, float32  # Correctly import float32 for usage
from math import sqrt  # Import sqrt explicitly for use in the kernel
import numpy as np
from raytracer.geometry import Sphere, Plane  # Import the geometry classes

@cuda.jit
def render_kernel(scene_data, camera_data, output, width, height, samples_per_pixel):
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

    # Initialize hit color and distance
    color = cuda.local.array(3, dtype=float32)
    for i in range(3):
        color[i] = 0.0  # Default background color (black)
    closest_t = float('inf')

    # Intersection tests
    for obj in scene_data:
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
                    color[0] = obj[5]  # Red channel
                    color[1] = obj[6]  # Green channel
                    color[2] = obj[7]  # Blue channel
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
                    color[0] = obj[7]  # Red channel
                    color[1] = obj[8]  # Green channel
                    color[2] = obj[9]  # Blue channel

    # Write the color to the output
    output[y, x, 0] = color[0]
    output[y, x, 1] = color[1]
    output[y, x, 2] = color[2]

class Renderer:
    def __init__(self, width, height, samples_per_pixel=1):
        self.width = width
        self.height = height
        self.samples_per_pixel = samples_per_pixel

    def render(self, scene, camera):
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
        for obj in scene.objects:
            if isinstance(obj, Sphere):
                scene_data_np.append([
                    1,  # Type: Sphere
                    *obj.center,  # Center
                    obj.radius,  # Radius
                    *obj.material.color  # Color (RGB)
                ])
            elif isinstance(obj, Plane):
                scene_data_np.append([
                    2,  # Type: Plane
                    *obj.point,  # A point on the plane
                    *obj.normal,  # Normal vector
                    *obj.material.color  # Color (RGB)
                ])

            elif isinstance(obj, Plane):
                scene_data_np.append([
                    2,                     # Type identifier for Plane
                    *obj.point,            # Point on the plane (3 values)
                    0.0,                   # Padding for unused field
                    *obj.normal,           # Normal (3 values)
                    *obj.material.color    # Material color (3 values)
                ])
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        # Convert to NumPy array
        scene_data_np = np.array(scene_data_np, dtype=np.float32)
        d_scene = cuda.to_device(scene_data_np)

        # Define thread and grid sizes
        threads_per_block = (16, 16)
        blocks_per_grid_x = (self.width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self.height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch kernel
        render_kernel[blocks_per_grid, threads_per_block](
            d_scene, d_camera, d_output, self.width, self.height, self.samples_per_pixel
        )

        # Copy output back to host
        output = d_output.copy_to_host()
        return output
