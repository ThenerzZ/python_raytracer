import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from raytracer.ray import Ray


class Renderer:
    def __init__(self, width, height, max_depth=5, samples_per_pixel=1, mode='preview'):
        self.base_width = width
        self.base_height = height
        self.max_depth = max_depth
        self.samples_per_pixel = samples_per_pixel

        # Adjust for preview or high-quality mode
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

        # Compile the CUDA kernel
        self.kernel = self.cuda_kernel()
        self.render_kernel = self.kernel.get_function("render")

    def render(self, scene, camera):
        aspect_ratio = self.width / self.height
        pixels = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Prepare data for GPU
        flat_pixels = pixels.ravel()
        scene_data = self.prepare_scene(scene)
        camera_data = np.array([camera.position, camera.forward, camera.up, camera.right], dtype=np.float32)

        # Send data to GPU
        scene_buffer = drv.In(scene_data)
        camera_buffer = drv.In(camera_data)
        output_buffer = drv.Out(flat_pixels)

        # Launch CUDA kernel
        self.render_kernel(
            scene_buffer,
            np.int32(len(scene.objects)),
            camera_buffer,
            np.int32(self.width),
            np.int32(self.height),
            np.int32(self.samples_per_pixel),
            np.int32(self.max_depth),
            output_buffer,
            block=(16, 16, 1),
            grid=(self.width // 16, self.height // 16, 1),
        )

        # Reshape flat array back to image
        pixels = flat_pixels.reshape((self.height, self.width, 3))
        return np.clip(pixels, 0, 1)

    def prepare_scene(self, scene):
        # Convert scene objects into GPU-compatible data
        objects = []
        for obj in scene.objects:
            if isinstance(obj, Sphere):
                # Object type (0 for Sphere), position, radius, and material color
                objects.append([0, *obj.center, obj.radius, *obj.material.color])
            # Add other object types (e.g., planes, boxes) as needed

        return np.array(objects, dtype=np.float32)

    def cuda_kernel(self):
        return SourceModule("""
        #include <math.h>

        __device__ float3 normalize(float3 v) {
            float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            return make_float3(v.x / length, v.y / length, v.z / length);
        }

        __device__ float dot(float3 a, float3 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        __device__ float3 reflect(float3 dir, float3 normal) {
            return dir - 2.0f * dot(dir, normal) * normal;
        }

        __global__ void render(
            float *scene_data, int num_objects,
            float *camera_data,
            int width, int height,
            int samples_per_pixel, int max_depth,
            float *output
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            int idx = (y * width + x) * 3;

            // Compute ray direction
            float u = (float(x) / width - 0.5f) * 2.0f;
            float v = (0.5f - float(y) / height) * 2.0f;

            float3 origin = make_float3(camera_data[0], camera_data[1], camera_data[2]);
            float3 direction = normalize(make_float3(u, v, -1.0f));

            // Initialize color
            float3 color = make_float3(0.0f, 0.0f, 0.0f);

            for (int sample = 0; sample < samples_per_pixel; sample++) {
                // Basic ray tracing loop (example for spheres)
                float t_min = 1e-4f;
                float t_max = 1e30f;

                // Iterate over objects
                for (int i = 0; i < num_objects; i++) {
                    float3 center = make_float3(scene_data[i * 7 + 1], scene_data[i * 7 + 2], scene_data[i * 7 + 3]);
                    float radius = scene_data[i * 7 + 4];
                    float3 oc = origin - center;
                    float a = dot(direction, direction);
                    float b = dot(oc, direction) * 2.0f;
                    float c = dot(oc, oc) - radius * radius;
                    float discriminant = b * b - 4 * a * c;

                    if (discriminant > 0) {
                        float t = (-b - sqrt(discriminant)) / (2.0f * a);
                        if (t > t_min && t < t_max) {
                            t_max = t;

                            // Shading (diffuse only for now)
                            float3 hit_point = origin + t * direction;
                            float3 normal = normalize(hit_point - center);
                            float3 light_dir = normalize(make_float3(1, 1, -1)); // Hardcoded light
                            float diffuse = fmaxf(dot(normal, light_dir), 0.0f);

                            color += diffuse * make_float3(scene_data[i * 7 + 5], scene_data[i * 7 + 6], scene_data[i * 7 + 7]);
                        }
                    }
                }
            }

            // Average samples and write to output
            color /= samples_per_pixel;
            output[idx] = color.x;
            output[idx + 1] = color.y;
            output[idx + 2] = color.z;
        }
        """)
