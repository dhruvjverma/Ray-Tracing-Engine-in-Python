from dataclasses import dataclass
from typing import Union
import math

@dataclass(frozen=True)
class Vector:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector") -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented

        return Vector(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    
    def __sub__(self, other: "Vector") -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented

        return Vector(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )
    
    def __mul__(self, scalar: Union[int, float]) -> "Vector":
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        return Vector(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )
    
    def __rmul__(self, scalar: Union[int, float]) -> "Vector":
        # Handles reverse multiplication (eg. 2 * Vector) by deferring to `__mul__`
        return self.__mul__(scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector":
        mag = self.magnitude()
        if mag == 0.0:
            return Vector(0.0, 0.0, 0.0) # Prevents ZeroDivisionError: A zero vector remains zero. This took a lot of time to find.

        return Vector(
            self.x / mag,
            self.y / mag,
            self.z / mag,
        )

    def __matmul__(self, other: "Vector") -> float:
        if not isinstance(other, Vector):
            return NotImplemented

        return (
            self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )
