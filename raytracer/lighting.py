class Light:
    def __init__(self, position, color, intensity):
        self.position = position
        self.color = color
        self.intensity = intensity

    def to_dict(self):
        return {
            "position": self.position,
            "color": self.color,
            "intensity": self.intensity
        }
