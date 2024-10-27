import numpy as np

class Vector:
    def __init__(self, a, b, polar=False):
        if polar:
            self.magnitude = a
            self.angle = b
            self.update_cartesian()
        else:
            self.x = a
            self.y = b
            self.update_polar()
    def update_polar(self):
        self.magnitude = np.sqrt(self.x**2 + self.y**2)
        self.angle = np.arctan2(self.y, self.x)

    def update_cartesian(self):
        self.x = self.magnitude * np.cos(self.angle)
        self.y = self.magnitude * np.sin(self.angle)
