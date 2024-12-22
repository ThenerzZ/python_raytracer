from raytracer.scene import Scene
from raytracer.geometry import Sphere, Plane
from raytracer.materials import Material
from raytracer.lighting import Light
from raytracer.camera import Camera
from raytracer.renderer import Renderer

def main():
    # Initialize the scene
    scene = Scene()

    # Add objects (e.g., sphere and planes)
    sphere_material = Material(color=[0.0, 0.0, 1.0], shininess=32)
    scene.add(Sphere(center=[0, 0, -5], radius=1, material=sphere_material))

    wall_material = Material(color=[0.5, 0.5, 0.5], shininess=8)
    scene.add(Plane(point=[0, -1, 0], normal=[0, 1, 0], material=wall_material))  # Floor
    scene.add(Plane(point=[0, 1, 0], normal=[0, -1, 0], material=wall_material))  # Ceiling
    scene.add(Plane(point=[-5, 0, 0], normal=[1, 0, 0], material=wall_material))  # Left wall
    scene.add(Plane(point=[5, 0, 0], normal=[-1, 0, 0], material=wall_material))  # Right wall
    scene.add(Plane(point=[0, 0, -10], normal=[0, 0, 1], material=wall_material))  # Back wall

    # Add a light source
    light = Light(position=[0, 0, 3], color=[1.0, 1.0, 1.0], intensity=50.0)
    scene.add(light.to_dict())

    # Set up the camera
    camera = Camera(
        position=[0, 0, 2],
        look_at=[0, 0, -5],
        up=[0, 1, 0],
        fov=60,
        aspect_ratio=16 / 9
    )

    # Create the renderer
    renderer = Renderer(width=800, height=450, samples_per_pixel=10)

    # Render the scene
    image = renderer.render(scene, camera)

    # Display the image
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
