from raytracer.camera import Camera
from raytracer.renderer import Renderer
from raytracer.scene import Scene
from raytracer.geometry import Sphere, Box
from raytracer.lighting import Light
from raytracer.materials import Material
import matplotlib.pyplot as plt


def main(mode='preview'):
    # Create the scene
    scene = Scene()

    # Materials
    room_material = Material(color=[0.3, 0.3, 0.3], reflectivity=0)  # Darker gray for walls, ceiling, and floor
    sphere_material = Material(color=[0.1, 0.1, 0.8], reflectivity=0.5)  # Shiny blue sphere

    # Add objects to the scene
    # Floor
    floor = Box(min_point=[-5, 0, -5], max_point=[5, 0.1, 5], material=room_material)
    scene.add_object(floor)

    # Walls
    left_wall = Box(min_point=[-5, 0, -5], max_point=[-4.9, 5, 5], material=room_material)
    right_wall = Box(min_point=[4.9, 0, -5], max_point=[5, 5, 5], material=room_material)
    back_wall = Box(min_point=[-5, 0, -5], max_point=[5, 5, -4.9], material=room_material)
    ceiling = Box(min_point=[-5, 4.9, -5], max_point=[5, 5, 5], material=room_material)

    scene.add_object(left_wall)
    scene.add_object(right_wall)
    scene.add_object(back_wall)
    scene.add_object(ceiling)

    # Sphere
    sphere = Sphere(center=[0, 1, 0], radius=1, material=sphere_material)
    scene.add_object(sphere)

    # Add a light source behind the camera
    light = Light(position=[0, 3, 5], intensity=3.0)  # Adjusted intensity and position
    scene.add_light(light)

    # Build BVH for optimization
    scene.build_bvh()

    # Create the camera inside the room
    camera = Camera(
        position=[0, 2, 4],  # Positioned inside the room
        look_at=[0, 1, 0],   # Looking at the sphere
        up=[0, 1, 0],        # Up direction
        fov=60,              # Field of view
        aspect_ratio=16 / 9
    )

    # Create the renderer
    renderer = Renderer(width=800, height=450, max_depth=3, samples_per_pixel=16, mode=mode)

    # Render the scene
    image = renderer.render(scene, camera)

    # Display the rendered image
    plt.figure(figsize=(10, 5))
    plt.title(f"Render Mode: {mode}")
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main(mode='preview')  # Render the room with a sphere and light source