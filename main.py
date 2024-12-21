from raytracer.camera import Camera
from raytracer.renderer import Renderer
from raytracer.scene import Scene
from raytracer.geometry import Sphere, Plane
from raytracer.lighting import Light
from raytracer.materials import Material
from raytracer.utils import checkerboard_texture

def main():
    # Create the scene
    scene = Scene()

    # Materials
    sphere_material = Material(color=[0.8, 0.1, 0.1], reflectivity=0.6)  # Shiny red sphere
    floor_material = Material(color=[0.5, 0.5, 0.5], texture=checkerboard_texture)  # Checkerboard floor
    wall_material = Material(color=[0.8, 0.8, 0.8])  # Neutral gray walls

    # Add objects to the scene
    floor = Plane(point=[0, 0, 0], normal=[0, 1, 0], material=floor_material)
    sphere = Sphere(center=[0, 1, -2], radius=1, material=sphere_material)
    back_wall = Plane(point=[0, 0, -5], normal=[0, 0, 1], material=wall_material)

    scene.add_object(floor)
    scene.add_object(sphere)
    scene.add_object(back_wall)

    # Add a light source
    light = Light(position=[0, 5, 5], intensity=2.0)
    scene.add_light(light)

    # Build BVH for optimization
    scene.build_bvh()

    # Create the camera
    camera = Camera(
        position=[0, 2, 8],  # Positioned to capture the entire scene
        look_at=[0, 1, 0],   # Looking at the sphere
        up=[0, 1, 0],        # Up direction
        fov=60,              # Field of view
        aspect_ratio=16 / 9
    )

    # Create the renderer
    renderer = Renderer(width=800, height=450, max_depth=3, samples_per_pixel=4)

    # Render the scene
    image = renderer.render(scene, camera)

    # Display the rendered image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
