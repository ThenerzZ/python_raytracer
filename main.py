from raytracer.camera import Camera
from raytracer.renderer import Renderer
from raytracer.scene import Scene
from raytracer.geometry import Sphere, Plane
from raytracer.lighting import Light
from raytracer.materials import Material
from raytracer.utils import checkerboard_texture

def main():
    # Adjusted Camera
    camera = Camera(
        position=[0, 2, 4],  # Closer to the sphere
        look_at=[0, 1, 0],   # Look directly at the sphere
        up=[0, 1, 0],
        fov=45,              # Narrower field of view for better focus
        aspect_ratio=16 / 9,
    )

    # Create Scene
    scene = Scene()

    # Materials
    reflective_material = Material(color=[0.8, 0.1, 0.1], reflectivity=0.8)  # Shiny red sphere
    floor_material = Material(color=[0.5, 0.5, 0.5], texture=checkerboard_texture)  # Checkerboard floor
    wall_material = Material(color=[0.8, 0.8, 0.8])  # Neutral gray walls
    ceiling_material = Material(color=[0.9, 0.9, 0.9])  # Bright white ceiling

    # Objects (Room and Sphere)
    # Floor
    floor = Plane(point=[0, 0, 0], normal=[0, 1, 0], material=floor_material)
    # Ceiling
    ceiling = Plane(point=[0, 5, 0], normal=[0, -1, 0], material=ceiling_material)
    # Back Wall
    back_wall = Plane(point=[0, 0, -5], normal=[0, 0, 1], material=wall_material)
    # Left Wall
    left_wall = Plane(point=[-5, 0, 0], normal=[1, 0, 0], material=wall_material)
    # Right Wall
    right_wall = Plane(point=[5, 0, 0], normal=[-1, 0, 0], material=wall_material)
    # Sphere
    sphere = Sphere(center=[0, 1, -1], radius=1, material=reflective_material)

    # Add Objects to Scene
    scene.add_object(floor)
    scene.add_object(ceiling)
    scene.add_object(back_wall)
    scene.add_object(left_wall)
    scene.add_object(right_wall)
    scene.add_object(sphere)

    # Light Source
    light = Light(position=[0, 4, 2], intensity=3.0)  # Above the sphere
    scene.add_light(light)

    # Render the Scene
    renderer = Renderer(width=1920, height=1080, max_depth=10)  # High resolution
    image = renderer.render(scene, camera)

    # Display the Rendered Image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()


