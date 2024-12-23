class Scene:
    def __init__(self):
        self.objects = []  # Store objects (e.g., spheres, boxes, planes)
        self.lights = []   # Store light sources

    def add(self, item):
        """
        Add an object or light source to the scene.
        :param item: Can be an object (Sphere, Box, Plane) or a Light
        """
        if isinstance(item, dict):  # Assuming lights are serialized as dictionaries
            self.lights.append(item)
        else:
            self.objects.append(item)

    def get_objects(self):
        """
        Retrieve all objects in the scene.
        :return: List of objects
        """
        return self.objects

    def get_lights(self):
        """
        Retrieve all lights in the scene.
        :return: List of lights
        """
        return self.lights
