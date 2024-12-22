import numpy as np

class Material:
    def __init__(self, color, reflectivity=0, transparency=0, refractive_index=1, shininess=50, texture=None):
        """
        Initialize a material with properties like color, reflectivity, and texture.

        Parameters:
        color (list): RGB color [r, g, b].
        reflectivity (float): Reflectivity factor (0 to 1).
        transparency (float): Transparency factor (0 to 1).
        refractive_index (float): Index of refraction for transparent materials.
        shininess (float): Shininess for specular highlights.
        texture (function): A texture function to override the base color.
        """
        self.color = np.array(color)
        self.reflectivity = reflectivity
        self.transparency = transparency
        self.refractive_index = refractive_index
        self.shininess = shininess
        self.texture = texture

    def get_color(self, point):
        """
        Get the color of the material at a specific point.

        Parameters:
        point (list): 3D point [x, y, z].

        Returns:
        list: RGB color [r, g, b].
        """
        if self.texture:
            return np.array(self.texture(point))
        return self.color
