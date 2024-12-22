Python Ray Tracer

A simple yet extensible ray tracing engine written in Python. This project showcases fundamental ray tracing concepts, including realistic lighting, shadows, reflections, and refraction. The engine renders a 3D scene with objects, materials, and light sources into a 2D image.

Features
Realistic Lighting: Supports diffuse and specular lighting with soft shadows.
Reflections and Refraction: Simulates shiny and transparent materials for realistic rendering.
Ambient Occlusion: Adds depth and realism by simulating indirect lighting in shadowed areas.
Customizable Materials: Easily define object materials with attributes like color, reflectivity, transparency, and shininess.
Basic Geometry: Supports spheres, planes, and boxes. Extensible for additional shapes.
BVH Optimization: Uses Bounding Volume Hierarchies (BVH) for efficient ray-object intersection tests.
Multi-Sampling Anti-Aliasing: Smoothens edges by averaging multiple rays per pixel.
Gamma Correction: Ensures the final image has realistic brightness and contrast.


Getting Started:

Prerequisites
Python 3.8 or higher

Libraries:
numpy
matplotlib

To install the required libraries:
pip install numpy matplotlib

Installation

Clone the repository:
git clone https://github.com/your-username/python-ray-tracer.git
cd python-ray-tracer

Run the main script:
python main.py

Project Structure

├── main.py                   # Entry point for the ray tracer
├── raytracer
│   ├── camera.py             # Camera class for generating rays
│   ├── renderer.py           # Main rendering engine
│   ├── scene.py              # Scene class to manage objects and lights
│   ├── ray.py                # Ray class
│   ├── geometry.py           # Geometric objects (Sphere, Box, etc.)
│   ├── lighting.py           # Light sources
│   ├── materials.py          # Material properties
│   ├── utils.py              # Utility functions (e.g., textures)
├── README.md                 # Project documentation


How It Works

Scene Setup:

Define a scene by adding objects (e.g., spheres, boxes) and light sources.
Customize materials for objects, including color, reflectivity, and transparency.
Rendering:

The Renderer generates rays for each pixel, computes intersections, and calculates lighting.
Supports advanced features like reflections, refraction, and ambient occlusion.
Output:

The final rendered image is displayed using matplotlib.
Features in Detail
Lighting
Supports diffuse lighting (Lambertian reflection) and specular highlights (Phong model).
Soft shadows are achieved by jittering shadow rays to simulate area light sources.
Reflections and Refraction
Realistic reflections for shiny materials.
Refraction for transparent materials like glass, with support for custom refractive indices.
Ambient Occlusion
Simulates indirect lighting by darkening occluded areas.
Anti-Aliasing
Multi-sampling smoothens edges by averaging colors of rays jittered within each pixel.
Gamma Correction
Corrects image brightness for realistic tone mapping.
Example Scene
The default scene renders a room with:

Gray walls, ceiling, and floor
A shiny blue sphere in the center
A single light source illuminating the room
You can customize the scene by modifying main.py.

Future Improvements
Add support for complex 3D models (e.g., OBJ file loading).
Implement more advanced materials like metallic and subsurface scattering.
Add global illumination for realistic indirect lighting.
Contributing
Contributions are welcome! If you have suggestions or find bugs, feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or feedback, reach out to:

Name: Fabian Wegner
Email: thenerz_lunix@proton.me
GitHub: ThenerzZ
