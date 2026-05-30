import math

from typing import Optional
from dataclasses import dataclass
from src.vector_math import Vector

@dataclass
class Ray:
    origin: Vector
    direction: Vector

    def __post_init__(self) -> None:
        # Automatically normalize the direction vector
        self.direction = self.direction.normalize()

@dataclass
class Sphere:
    center: Vector
    radius: float
    color: tuple[int, int, int]

    def intersect(self, ray: Ray) -> Optional[float]:
        """
        Returns the distance to the nearest intersection point,
        or None if the ray misses the Sphere.
        """

        of = ray.origin - self.center

        # Optimization: Since ray.direction is normalized, 'a' is always 1.0.
        # I'll completely omit 'a' from the quadratic equation. This improves performance
        b = 2.0 * (of @ ray.direction)
        c = (of @ of) - self.radius**2

        discriminant = b**2 - 4.0 * c # removed `a` from here

        if discriminant < 0.0:
            return None

        sqrt_discriminant = math.sqrt(discriminant)

        t1 = (-b - sqrt_discriminant) / 2.0 # removed `a` from here
        t2 = (-b + sqrt_discriminant) / 2.0 # removed `a` from here

        # Guard against Shadow Acne using a small epsilon value instead of 0.0
        EPSILON = 1e-4

        # Keep only intersections in front of the ray
        hits = [t for t in (t1, t2) if t > EPSILON]

        if not hits:
            return None

        return min(hits)