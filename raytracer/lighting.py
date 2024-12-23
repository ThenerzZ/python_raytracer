import numpy as np

class Light:
    def __init__(self, position, color, intensity=1.0):
        """
        Initialize a light source.
        :param position: Position of the light [x, y, z]
        :param color: Color of the light [r, g, b]
        :param intensity: Intensity of the light
        """
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.intensity = intensity

    def to_dict(self):
        """
        Serialize light to a dictionary for kernel use.
        """
        return {
            "position": self.position.tolist(),
            "color": self.color.tolist(),
            "intensity": self.intensity,
        }

def calculate_lighting(hit_point, normal, view_dir, lights, material):
    """
    Calculate the lighting at a point using the Phong reflection model.
    :param hit_point: The point of intersection [x, y, z]
    :param normal: The normal vector at the intersection [x, y, z]
    :param view_dir: Direction to the camera/viewer [x, y, z]
    :param lights: List of Light objects
    :param material: Material properties (e.g., color, shininess, metallic, roughness)
    :return: Final color at the intersection
    """
    ambient = 0.1  # Ambient lighting constant
    diffuse_color = material.color
    specular_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # White specular highlights
    shininess = material.shininess

    final_color = ambient * diffuse_color  # Start with ambient contribution

    for light in lights:
        light_dir = light.position - hit_point
        light_distance = np.linalg.norm(light_dir)
        light_dir /= light_distance  # Normalize

        # Attenuation based on distance (inverse square law)
        attenuation = light.intensity / (light_distance * light_distance)

        # Diffuse shading
        diffuse_intensity = max(np.dot(normal, light_dir), 0.0)
        diffuse = attenuation * diffuse_intensity * light.color * diffuse_color

        # Specular shading (Blinn-Phong model)
        halfway_dir = (view_dir + light_dir) / np.linalg.norm(view_dir + light_dir)
        spec_intensity = max(np.dot(normal, halfway_dir), 0.0) ** shininess
        specular = attenuation * spec_intensity * light.color * specular_color

        # Add contributions to final color
        final_color += diffuse + specular

    # Clamp color to [0, 1]
    return np.clip(final_color, 0, 1)
