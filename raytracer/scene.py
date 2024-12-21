from raytracer.camera import Camera
from raytracer.renderer import Renderer
from raytracer.lighting import Light
from raytracer.geometry import Sphere, Plane
from raytracer.materials import Material
from raytracer.utils import checkerboard_texture

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

    def intersect(self, ray):
        closest_t = float('inf')
        hit_object = None
        for obj in self.objects:
            t = obj.intersect(ray)
            if t and t < closest_t:
                closest_t = t
                hit_object = obj
        return hit_object, closest_t

def main():
    # Create camera
    camera = Camera(position=[0, 1, 3], look_at=[0, 0, 0], up=[0, 1, 0], fov=60, aspect_ratio=16/9)

    # Create scene
    scene = Scene()

    # Materials
    red_material = Material(color=[1, 0, 0], reflectivity=0.5)
    blue_material = Material(color=[0, 0, 1], reflectivity=0.3)
    ground_material = Material(color=[0.8, 0.8, 0.8], texture=checkerboard_texture)

    # Objects
    sphere1 = Sphere(center=[-0.5, 0.5, -3], radius=0.5, material=red_material)
    sphere2 = Sphere(center=[1, 0.5, -4], radius=0.5, material=blue_material)
    ground = Plane(point=[0, 0, 0], normal=[0, 1, 0], material=ground_material)

    scene.add_object(sphere1)
    scene.add_object(sphere2)
    scene.add_object(ground)

    # Light
    light = Light(position=[5, 5, 5], intensity=1.5)
    scene.add_light(light)

    # Render
    renderer = Renderer(width=800, height=450)
    image = renderer.render(scene, camera)

    # Display
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
