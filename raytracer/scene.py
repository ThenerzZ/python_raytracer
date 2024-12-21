from raytracer.geometry import BVHNode

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.bvh = None

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

    def build_bvh(self):
        if self.objects:
            self.bvh = BVHNode(self.objects)

    def intersect(self, ray):
        if self.bvh is None:
            raise ValueError("BVH is not built. Call `build_bvh` before rendering.")
        return self.bvh.intersect(ray)

