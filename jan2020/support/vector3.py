import math


class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def to_string(self):
        return "x: {0}, y:{1}, z:{2}".format(self.x, self.y, self.z)

    def clone(self):
        return Vector3(self.x, self.y, self.z)

    def set_x(self, x):
        self.x = x
    
    def get_x(self):
        return self.x

    def set_y(self, y):
        self.y = y
    
    def get_y(self):
        return self.y

    def set_z(self, z):
        self.z = z

    def get_z(self):
        return self.z 

    def set_components_from_array(self, pointList):
        self.x = pointList[0]
        self.y = pointList[1]
        self.z = pointList[2]

    def get_components(self):
        return self.x, self.y, self.z

    def add(self, v):
        self.x += v.x
        self.y += v.y
        self.z += v.z
        return self

    def sub(self, v):
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
        return self

    def multiply(self, v):
        self.x *= v.x
        self.y *= v.y
        self.z *= v.z
        return self
    
    def divide(self, v):
        self.x /= v.x
        self.y /= v.y
        self.z /= v.z
        return self

    def add_scaler(self, s):
        self.x += s
        self.y += s
        self.z += s
        return self

    def sub_scaler(self, s):
        self.x -= s
        self.y -= s
        self.z -= s
        return self

    def add_vectors(self, a, b):
        self.x = a.x + b.x
        self.y = a.y + b.y
        self.z = a.z + b.z 
        return self

    def sub_vectors(self, a, b):
        self.x = a.x - b.x
        self.y = a.y - b.y
        self.z = a.z - b.z 
        return self

    def add_scaled_vector(self, v, s):
        self.x += v.x * s
        self.y += v.y * s
        self.z += v.z * s 
        return self

    def multiply_scalar(self, scalar):
        if math.isfinite(scalar):
            self.x *= scalar
            self.y *= scalar
            self.z *= scalar
        else:
            self.x = 0
            self.y = 0
            self.z = 0
        return self

    def divide_scalar(self, scalar):
        return self.multiply_scalar(1 / scalar)
    
    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v):
        x = self.x
        y = self.y
        z = self.z

        self.x = y * v.z - z * v.y
        self.y = z * v.x - x * v.z
        self.z = x * v.y - y * v.x
        return self

    def length_sq(self):
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        return self.divide_scalar(self.length())

    def set_length(self, length):
        return self.multiply_scalar(length / self.length())

    def lerp(self, v, alpha):
        self.x += (v.x - self.x) * alpha
        self.y += (v.y - self.y) * alpha
        self.z += (v.z - self.z) * alpha
        return self

    def angle_to(self, v):
        theta = self.dot( v ) / math.sqrt( self.length_sq() * v.length_sq() )

        # // clamp, to handle numerical problems

        return math.acos( math.clamp( theta, - 1, 1 ) )
    
    def distance_to(self, v):
        return math.sqrt( self.distance_to_squared( v ) )

    def distance_to_squared(self, v):
        dx = self.x - v.x
        dy = self.y - v.y
        dz = self.z - v.z
        return dx * dx + dy * dy + dz * dz

    def to_serializable(self):
        self

    @staticmethod
    def from_deserialized(deserialized_vector_3):
        return Vector3(
            deserialized_vector_3['x'],
            deserialized_vector_3['y'],
            deserialized_vector_3['z'],
        )
