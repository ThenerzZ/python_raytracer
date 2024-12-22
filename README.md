---

# **Basic Python Ray Tracer**

This is a foundational **ray tracing engine** implemented in Python. It demonstrates core ray tracing principles, including rendering simple 3D objects, basic lighting, and shadows.

---

## **Features**

- **Basic Lighting**: Supports diffuse and specular lighting.
- **Shadows**: Hard shadows using point light sources.
- **Simple Geometry**: Includes spheres, planes, and boxes.
- **Custom Materials**: Objects can have colors and reflectivity.

---

## **Getting Started**

### **Prerequisites**

Ensure you have Python 3.8 or higher and install the required libraries:

```bash
pip install numpy matplotlib
```

---

### **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/python-ray-tracer.git
   cd python-ray-tracer
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

---

## **How It Works**

1. **Scene Setup**:
   - Add objects (e.g., spheres, planes) to the scene.
   - Define a single light source to illuminate the scene.

2. **Rendering**:
   - Rays are cast from the camera, interacting with objects to compute colors based on material and lighting.

3. **Output**:
   - The final rendered image is displayed using `matplotlib`.

---

## **Project Structure**

```
.
├── main.py                   # Entry point of the project
├── raytracer/
│   ├── camera.py             # Camera class for generating rays
│   ├── renderer.py           # Basic rendering logic
│   ├── scene.py              # Manages objects and lights
│   ├── ray.py                # Defines the Ray class
│   ├── geometry.py           # Geometric objects like Sphere and Plane
│   ├── lighting.py           # Defines point lights
│   ├── materials.py          # Basic material properties
```

---

## **Example Scene**

The default scene renders:
- A simple room 
- A single sphere in the center.
- A point light source illuminating the sphere.

---

## **Limitations**

- No soft shadows, reflections, or advanced features.
- Lighting and material properties are basic.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
