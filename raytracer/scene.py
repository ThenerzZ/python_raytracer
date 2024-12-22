from raytracer.geometry import BVHNode

class Scene:
    def __init__(self):
        """
        Initialize the scene with empty lists for objects and lights.
        A BVH (Bounding Volume Hierarchy) is used for efficient ray intersection.
        """
        self.objects = []  # List to store geometric objects (e.g., Sphere, Plane)
        self.lights = []   # List to store light sources
        self.bvh = None    # BVH structure for optimized intersection

    def add(self, obj):
        """
        Adds an object to the scene (either a geometric object or a light source).

        Parameters:
        obj (object): The object to add (e.g., Sphere, Plane, or Light).
        """
        if isinstance(obj, dict) and "position" in obj and "color" in obj and "intensity" in obj:
            # If obj has light attributes, treat it as a light source
            self.add_light(**obj)
        else:
            # Otherwise, treat it as a geometric object
            self.add_object(obj)

    def add_object(self, obj):
        """
        Adds a geometric object to the scene.

        Parameters:
        obj (object): The geometric object to add.
        """
        self.objects.append(obj)

    def add_light(self, position, color, intensity):
        """
        Adds a light source to the scene.

        Parameters:
        position (list): Light position [x, y, z].
        color (list): Light color [r, g, b].
        intensity (float): Light intensity.
        """
        self.lights.append({
            'position': position,
            'color': color,
            'intensity': intensity
        })

    def build_bvh(self):
        """
        Builds the Bounding Volume Hierarchy (BVH) for efficient ray intersections.
        """
        if self.objects:
            self.bvh = BVHNode(self.objects)

    def intersect(self, ray):
        """
        Intersects a ray with the scene's objects using the BVH.

        Parameters:
        ray (Ray): The ray to test for intersection.

        Returns:
        Intersection: The closest intersection, or None if no intersection occurs.
        """
        if self.bvh is None:
            raise ValueError("BVH is not built. Call `build_bvh` before rendering.")
        return self.bvh.intersect(ray)

    def get_objects(self):
        """
        Returns the list of objects in the scene.

        Returns:
        list: The geometric objects in the scene.
        """
        return self.objects

    def get_lights(self):
        """
        Returns the list of light sources in the scene.

        Returns:
        list: The light sources in the scene.
        """
        return self.lights
