from numba import cuda
import numpy as np


@cuda.jit
def render_kernel(scene_data, camera_data, output, width, height, samples_per_pixel):
    x, y = cuda.grid(2)

    if x >= width or y >= height:
        return

    idx = (y * width + x) * 3  # Flattened index
    u = (x + 0.5) / width
    v = (y + 0.5) / height

    # Camera setup
    origin = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        origin[i] = camera_data[i]

    direction = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        direction[i] = (
            camera_data[3 + i] + u * camera_data[6 + i] + v * camera_data[9 + i]
        )

    # Normalize direction
    magnitude = 0.0
    for i in range(3):
        magnitude += direction[i] * direction[i]
    magnitude = cuda.sqrt(magnitude)
    for i in range(3):
        direction[i] /= magnitude

    # Basic ray direction-based coloring for testing
    output[y, x, 0] = abs(direction[0])  # Red channel
    output[y, x, 1] = abs(direction[1])  # Green channel
    output[y, x, 2] = abs(direction[2])  # Blue channel


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

        # Camera and scene data
        camera_data = np.hstack(
            [camera.position, camera.forward, camera.right, camera.up]
        ).astype(np.float32)
        d_camera = cuda.to_device(camera_data)

        # Placeholder for scene data
        scene_data = np.array(scene.objects, dtype=np.float32)  # Example placeholder
        d_scene = cuda.to_device(scene_data)

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
