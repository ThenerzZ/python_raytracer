import matplotlib.pyplot as plt
from raytracer.renderer import Renderer
from raytracer.camera import Camera
from raytracer.scene import Scene
from raytracer.geometry import Sphere, Plane
from raytracer.material import Material


def main():
    # Renderer settings
    width, height = 800, 450
    samples_per_pixel = 16

    # Set up the scene
    scene = Scene()

    # Materials
    grey_wall = Material(color=[0.5, 0.5, 0.5])  # Grey walls
    blue_sphere = Material(color=[0.0, 0.0, 1.0])  # Blue sphere

    # Add objects to the scene
    scene.add(Sphere(center=[0, 0, -5], radius=1, material=blue_sphere))  # Sphere in the center of the room
    scene.add(Plane(point=[0, -1, 0], normal=[0, 1, 0], material=grey_wall))  # Floor
    scene.add(Plane(point=[0, 1, 0], normal=[0, -1, 0], material=grey_wall))  # Ceiling
    scene.add(Plane(point=[0, 0, -10], normal=[0, 0, 1], material=grey_wall))  # Back wall
    scene.add(Plane(point=[-5, 0, 0], normal=[1, 0, 0], material=grey_wall))  # Left wall
    scene.add(Plane(point=[5, 0, 0], normal=[-1, 0, 0], material=grey_wall))  # Right wall

    # Lighting
    scene.add_light(position=[0, 2, 0], color=[1.0, 1.0, 1.0], intensity=2.0)  # Light source above the room

    # Set up the camera
    camera = Camera(
        position=[0, 0, 5],  # Camera position in the room
        forward=[0, 0, -1],  # Looking towards the sphere
        right=[1, 0, 0],     # Camera's right direction
        up=[0, 1, 0]         # Camera's up direction
    )

    # Create the renderer
    renderer = Renderer(width, height, samples_per_pixel)

    # Render the scene
    image = renderer.render(scene, camera)

    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
