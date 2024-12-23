import numpy as np

class Material:
    def __init__(self, color, shininess=32, reflectivity=0.0, transparency=0.0, refractive_index=1.0):
        """
        Represents the material properties of an object.

        :param color: Base color of the material (r, g, b)
        :param shininess: Shininess factor for specular highlights
        :param reflectivity: Reflectivity of the material (0.0 to 1.0)
        :param transparency: Transparency of the material (0.0 to 1.0)
        :param refractive_index: Refractive index for light bending (used for transparency)
        """
        self.color = np.array(color, dtype=np.float32)
        self.shininess = shininess
        self.reflectivity = reflectivity
        self.transparency = transparency
        self.refractive_index = refractive_index

    def get_color(self):
        """Returns the base color of the material."""
        return self.color

    def get_shininess(self):
        """Returns the shininess factor."""
        return self.shininess

    def get_reflectivity(self):
        """Returns the reflectivity of the material."""
        return self.reflectivity

    def get_transparency(self):
        """Returns the transparency of the material."""
        return self.transparency

    def get_refractive_index(self):
        """Returns the refractive index of the material."""
        return self.refractive_index

# Example predefined materials
MIRROR = Material(color=[1.0, 1.0, 1.0], shininess=128, reflectivity=0.9, transparency=0.0)
GLASS = Material(color=[1.0, 1.0, 1.0], shininess=64, reflectivity=0.1, transparency=0.9, refractive_index=1.5)
MATTE_RED = Material(color=[1.0, 0.0, 0.0], shininess=16, reflectivity=0.1, transparency=0.0)
MATTE_BLUE = Material(color=[0.0, 0.0, 1.0], shininess=16, reflectivity=0.1, transparency=0.0)
