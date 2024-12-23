from raytracer.scene import Scene
from raytracer.geometry import Sphere, Box  # Ensure Box is imported
from raytracer.materials import Material
from raytracer.lighting import Light
from raytracer.camera import Camera
from raytracer.renderer import Renderer
import matplotlib.pyplot as plt

def main():
    # Initialize the scene
    scene = Scene()

    # Define room dimensions
    room_size = 10.0
    wall_thickness = 0.1

    # Add walls (as boxes)
    wall_material = Material(color=[0.5, 0.5, 0.5], shininess=8)

    # Floor
    scene.add(Box(
        min_corner=[-room_size, -room_size, -room_size],
        max_corner=[room_size, -room_size + wall_thickness, room_size],
        material=wall_material
    ))

    # Ceiling
    scene.add(Box(
        min_corner=[-room_size, room_size - wall_thickness, -room_size],
        max_corner=[room_size, room_size, room_size],
        material=wall_material
    ))

    # Back wall
    scene.add(Box(
        min_corner=[-room_size, -room_size, -room_size],
        max_corner=[room_size, room_size, -room_size + wall_thickness],
        material=wall_material
    ))

    # Front wall (not visible but prevents light from escaping)
    scene.add(Box(
        min_corner=[-room_size, -room_size, room_size - wall_thickness],
        max_corner=[room_size, room_size, room_size],
        material=wall_material
    ))

    # Left wall
    scene.add(Box(
        min_corner=[-room_size, -room_size, -room_size],
        max_corner=[-room_size + wall_thickness, room_size, room_size],
        material=wall_material
    ))

    # Right wall
    scene.add(Box(
        min_corner=[room_size - wall_thickness, -room_size, -room_size],
        max_corner=[room_size, room_size, room_size],
        material=wall_material
    ))

    # Add a sphere in the center of the room
    sphere_material = Material(color=[0.0, 0.0, 1.0], shininess=32)
    scene.add(Sphere(center=[0, 0, 0], radius=1, material=sphere_material))

    # Add a light source behind the camera
    light = Light(position=[0, 0, 8], color=[1.0, 1.0, 1.0], intensity=50.0)
    scene.add(light.to_dict())

    # Initialize the camera
    camera = Camera(
        position=[0, 0, 8],
        look_at=[0, 0, 0],
        up=[0, 1, 0],
        fov=60,
        aspect_ratio=16 / 9
    )

    # Create the renderer
    renderer = Renderer(width=800, height=450, samples_per_pixel=10)

    # Render the scene
    image = renderer.render(scene, camera)

    # Display the rendered image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
