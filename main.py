from raytracer.renderer import Renderer
from raytracer.camera import Camera
from raytracer.scene import Scene
from raytracer.geometry import Sphere, Plane
from raytracer.materials import Material
import matplotlib.pyplot as plt

def main():
    width, height = 800, 450
    samples_per_pixel = 16

    # Create the scene
    scene = Scene()

    # Materials
    grey_material = Material(color=[0.5, 0.5, 0.5])  # Grey walls
    sphere_material = Material(color=[1.0, 0.0, 0.0])  # Red sphere

    # Add a sphere to the scene
    scene.add(Sphere(center=[0, 0, -5], radius=1, material=sphere_material))

    # Add walls to create a room
    scene.add(Plane(point=[0, -1, 0], normal=[0, 1, 0], material=Material(color=[0.5, 0.5, 0.5])))  # Floor
    scene.add(Plane(point=[0, 1, 0], normal=[0, -1, 0], material=Material(color=[0.5, 0.0, 0.0])))  # Ceiling
    scene.add(Plane(point=[-5, 0, 0], normal=[1, 0, 0], material=Material(color=[0.0, 0.5, 0.0])))  # Left wall
    scene.add(Plane(point=[5, 0, 0], normal=[-1, 0, 0], material=Material(color=[0.0, 0.0, 0.5])))  # Right wall
    scene.add(Plane(point=[0, 0, -10], normal=[0, 0, 1], material=Material(color=[0.5, 0.5, 0.0])))  # Back wall

    # Light source
    light_pos = [0, 2, -3]  # Light above and slightly in front of the sphere
    light_color = [1.0, 1.0, 1.0]  # White light
    light_intensity = 2.0

    # Camera setup
    camera = Camera(
        position=[0, 0, 2],  # Camera closer to the scene
        look_at=[0, 0, -5],   # Looking directly at the sphere
        up=[0, 1, 0],         # Maintain upright orientation
        fov=60,               # Reasonable FOV
        aspect_ratio=width / height
    )

    # Create the renderer
    renderer = Renderer(width, height, samples_per_pixel)

    # Render the scene
    image = renderer.render(scene, camera, light_pos, light_color, light_intensity)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
