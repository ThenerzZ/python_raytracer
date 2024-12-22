class Scene:
    def __init__(self):
        self.objects = []  # List to hold objects (e.g., spheres, planes)
        self.lights = []   # List to hold light sources

    def add(self, obj):
        """
        Add an object or light to the scene.
        If it's a dictionary with light attributes, add to lights.
        Otherwise, add to objects.
        """
        if isinstance(obj, dict) and "position" in obj and "color" in obj and "intensity" in obj:
            self.lights.append(obj)
        else:
            self.objects.append(obj)

    def get_objects(self):
        """
        Return the list of objects in the scene.
        """
        return self.objects

    def get_lights(self):
        """
        Return the list of lights in the scene.
        """
        return self.lights

    def serialize_objects(self):
        """
        Serialize the objects into a NumPy-compatible format.
        Each object must have its type, properties, and material serialized.
        """
        serialized_objects = []
        for obj in self.objects:
            if isinstance(obj, Sphere):
                # Serialize sphere: [type, center_x, center_y, center_z, radius, material_color_r, g, b, shininess]
                serialized_objects.append([1, *obj.center, obj.radius, *obj.material.color, obj.material.shininess])
            elif isinstance(obj, Plane):
                # Serialize plane: [type, point_x, point_y, point_z, normal_x, normal_y, normal_z, material_color_r, g, b, shininess]
                serialized_objects.append([2, *obj.point, *obj.normal, *obj.material.color, obj.material.shininess])
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        return np.array(serialized_objects, dtype=np.float32)

    def serialize_lights(self):
        """
        Serialize the lights into a NumPy-compatible format.
        Each light should have its position, color, and intensity serialized.
        """
        serialized_lights = []
        for light in self.lights:
            serialized_lights.extend(light["position"] + light["color"] + [light["intensity"]])
        return np.array(serialized_lights, dtype=np.float32)
