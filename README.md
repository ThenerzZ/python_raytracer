# **Python Ray Tracer**

Welcome to the **Python Ray Tracer**! This project is a lightweight and extensible ray tracing engine that renders realistic 3D scenes. From simulating reflections to soft shadows and ambient occlusion, this engine captures the fundamentals of ray tracing in Python.

---

## **Features**

✨ **Realistic Lighting**: Supports diffuse, specular, and soft shadows.  
✨ **Reflections and Refraction**: Handles shiny and transparent materials like glass.  
✨ **Ambient Occlusion**: Adds depth to shadowed regions.  
✨ **Anti-Aliasing**: Smoothens edges with multi-sampling.  
✨ **Gamma Correction**: Ensures realistic brightness and contrast.  
✨ **BVH Optimization**: Speeds up ray-object intersections.  
✨ **Customizable Materials**: Easily tweak colors, reflectivity, and more.

---

## **Getting Started**

### **Prerequisites**

Before you start, make sure you have:
- **Python 3.8 or higher**
- The following libraries installed:
  ```bash
  pip install numpy matplotlib
  ```

### **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/ThenerzZ/python_raytracer.git
   cd python-ray-tracer
   ```
2. Run the main script:
   ```bash
   python main.py
   ```

---

## **How It Works**

### **1. Scene Setup**
Define a scene by adding:
- Objects like spheres, planes, and boxes.
- Light sources with adjustable position and intensity.
- Materials with customizable properties like color, reflectivity, and transparency.

### **2. Rendering**
- The `Renderer` computes how rays interact with objects, materials, and lights.
- Advanced features like reflections, refraction, and ambient occlusion are supported.

### **3. Output**
- The rendered 3D scene is displayed as a 2D image using `matplotlib`.

---

## **Project Structure**

```
.
├── main.py                   # Entry point of the application
├── raytracer/
│   ├── camera.py             # Defines the camera for ray generation
│   ├── renderer.py           # Main rendering logic
│   ├── scene.py              # Manages objects and lights in the scene
│   ├── ray.py                # Ray class
│   ├── geometry.py           # Sphere, Box, and other shapes
│   ├── lighting.py           # Defines light sources
│   ├── materials.py          # Material definitions
│   ├── utils.py              # Utility functions (e.g., textures)
```

---

## **Features in Detail**

### **Lighting**
- Diffuse and specular lighting based on the Phong reflection model.
- Soft shadows using multiple jittered rays to simulate area lights.

### **Reflections and Refraction**
- Reflections allow for shiny surfaces like metals.
- Refraction simulates transparency, supporting custom refractive indices.

### **Ambient Occlusion**
- Adds realism by darkening corners and occluded areas.

### **Anti-Aliasing**
- Multi-sampling smoothens jagged edges for high-quality renders.

### **Gamma Correction**
- Corrects brightness to make the final image look more natural.

---

## **Example Scene**

The default scene renders:
- A gray room (walls, floor, and ceiling).
- A shiny blue sphere in the center.
- A light source behind the camera, illuminating the room.

Rendered output:

[Rendered Scene](**https://imgur.com/a/wa2LmHh**)  


---

## **Future Improvements**

Here are some exciting ideas for extending this project:
1. Add support for loading complex 3D models (e.g., OBJ files).
2. Implement advanced materials like metallic surfaces or subsurface scattering.
3. Introduce global illumination for realistic indirect lighting.

---

## **Contributing**

We’d love your help! If you’ve got an idea or spot a bug, feel free to:
- Open an **issue**.
- Submit a **pull request**.

---

## **License**

This project is licensed under the MIT License. Check the [LICENSE](LICENSE) file for details.

---

## **Contact**

Have questions or feedback?  
Reach out here:
- **Name**: Fabian Wegner  
- **Email**: thenerz_lunix@proton.me  
- **GitHub**: ThenerzZ

---

