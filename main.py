from raytracer.scene import Scene
from raytracer.camera import Camera
from raytracer.renderer import Renderer
from raytracer.geometry import Sphere, Box
from raytracer.materials import MIRROR, GLASS, Material
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Scene setup
    scene = Scene()

    # Add walls using boxes to form a 3D room
    floor_material = Material(color=[0.9, 0.9, 0.9], shininess=32, reflectivity=0.3)  # Light grey floor
    ceiling_material = Material(color=[0.7, 0.7, 0.9], shininess=32, reflectivity=0.3)  # Light blue ceiling
    wall_material_back = Material(color=[1.0, 1.0, 0.0], shininess=8, reflectivity=0.1)  # Yellow back wall
    wall_material_left = Material(color=[0.0, 1.0, 0.0], shininess=8, reflectivity=0.1)  # Green left wall
    wall_material_right = Material(color=[1.0, 0.0, 0.0], shininess=8, reflectivity=0.1)  # Red right wall

    # Define the room dimensions
    room_size = 5.0

    # Floor and ceiling
    scene.add(Box(min_corner=[-room_size, -room_size, -room_size], max_corner=[room_size, -room_size + 0.1, room_size], material=floor_material))
    scene.add(Box(min_corner=[-room_size, room_size - 0.1, -room_size], max_corner=[room_size, room_size, room_size], material=ceiling_material))

    # Walls
    scene.add(Box(min_corner=[-room_size, -room_size, -room_size], max_corner=[room_size, room_size, -room_size + 0.1], material=wall_material_back))  # Back wall
    scene.add(Box(min_corner=[-room_size, -room_size, -room_size], max_corner=[-room_size + 0.1, room_size, room_size], material=wall_material_left))  # Left wall
    scene.add(Box(min_corner=[room_size - 0.1, -room_size, -room_size], max_corner=[room_size, room_size, room_size], material=wall_material_right))  # Right wall

    # Add a reflective sphere in the center
    reflective_sphere_material = MIRROR
    scene.add(Sphere(center=[0, -room_size + 1.5, 0], radius=1, material=reflective_sphere_material))

    # Add a glass sphere near the right wall
    glass_sphere_material = GLASS
    scene.add(Sphere(center=[2, -room_size + 1.5, 2], radius=1, material=glass_sphere_material))

    # Add primary light source
    scene.add_light({
        "position": [0, room_size - 1, 0],  # Slightly lower light source
        "color": [1.0, 1.0, 1.0],
        "intensity": 200.0  # Reduced intensity for balanced illumination
    })

    # Add secondary light source
    scene.add_light({
        "position": [-2, room_size - 1, 2],
        "color": [0.5, 0.5, 0.5],
        "intensity": 100.0  # Secondary light for better ambient effect
    })

    # Camera setup
    camera = Camera(position=[0, 0, -20], forward=[0, 0, 1], up=[0, 1, 0], right=[1, 0, 0])

    # Renderer setup
    renderer = Renderer(width=800, height=600, samples_per_pixel=4)

    # Debugging scene objects and lights
    print("Scene Objects:")
    for obj in scene.get_objects():
        print(obj)

    print("Scene Lights:")
    for light in scene.get_lights():
        print(light)

    # Render the scene
    image = renderer.render(scene, camera)

    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
