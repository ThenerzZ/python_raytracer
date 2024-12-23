import numpy as np

class Camera:
    def __init__(self, position, forward, up, right):
        """
        Camera setup for the scene.

        :param position: The position of the camera [x, y, z]
        :param forward: The forward direction vector of the camera [x, y, z]
        :param up: The up direction vector of the camera [x, y, z]
        :param right: The right direction vector of the camera [x, y, z]
        """
        self.position = np.array(position, dtype=np.float32)
        self.forward = np.array(forward, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.right = np.array(right, dtype=np.float32)

    def get_position(self):
        return self.position

    def get_forward(self):
        return self.forward

    def get_up(self):
        return self.up

    def get_right(self):
        return self.right
