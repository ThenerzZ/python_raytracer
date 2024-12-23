import numpy as np

class Material:
    def __init__(self, color, shininess=32, metallic=0.0, roughness=0.5):
        """
        Initialize a material with PBR properties.
        :param color: Base color of the material [r, g, b]
        :param shininess: Shininess exponent for specular highlights
        :param metallic: Metallic property (0 = non-metal, 1 = pure metal)
        :param roughness: Roughness (0 = smooth, 1 = rough)
        """
        self.color = np.array(color, dtype=np.float32)
        self.shininess = shininess
        self.metallic = metallic
        self.roughness = roughness


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
