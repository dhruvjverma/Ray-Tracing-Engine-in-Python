import math
import numpy as np
from typing import Optional, Protocol, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Configures how a surface interacts with light rays."""
    diffuse: float = 1.0
    specular: float = 0.0
    shininess: int = 1


@dataclass
class Ray:
    origin: np.ndarray      # Shape (3,)
    direction: np.ndarray   # Shape (3,)

    def __post_init__(self) -> None:
        mag = np.linalg.norm(self.direction)
        if mag > 0:
            self.direction = self.direction / mag


class Intersectable(Protocol):
    color: Tuple[int, int, int]
    material: Material

    # Scalar interface — preserved for backward-compatibility and standalone testing
    def intersect(self, ray: Ray) -> Optional[float]: ...
    def get_normal(self, hit_point: np.ndarray) -> np.ndarray: ...

    # Batch interface — accepts whole-image (H, W, 3) grid tensors
    def intersect_batch(self, ray_origins: np.ndarray, ray_directions: np.ndarray) -> np.ndarray: ...
    def get_normal_batch(self, hit_points: np.ndarray) -> np.ndarray: ...


@dataclass
class Sphere:
    center: np.ndarray      # Shape (3,)
    radius: float
    color: Tuple[int, int, int]
    material: Material = Material()

    # --- Scalar Layer ---

    def intersect(self, ray: Ray) -> Optional[float]:
        of = ray.origin - self.center
        b = 2.0 * np.dot(of, ray.direction)
        c = np.dot(of, of) - self.radius**2
        discriminant = b**2 - 4.0 * c
        if discriminant < 0.0:
            return None
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / 2.0
        t2 = (-b + sqrt_discriminant) / 2.0
        EPSILON = 1e-4
        hits = [t for t in (t1, t2) if t > EPSILON]
        return min(hits) if hits else None

    def get_normal(self, hit_point: np.ndarray) -> np.ndarray:
        raw_normal = hit_point - self.center
        return raw_normal / np.linalg.norm(raw_normal)

    # --- Batch Layer ---

    def intersect_batch(self, ray_origins: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
        H, W, _ = ray_origins.shape
        EPSILON = 1e-4

        of = ray_origins - self.center                                  # (H, W, 3)

        a = np.sum(ray_directions * ray_directions, axis=-1)            # (H, W)
        b = 2.0 * np.sum(of * ray_directions, axis=-1)                  # (H, W)
        c = np.sum(of * of, axis=-1) - self.radius**2                   # (H, W)

        discriminant = b**2 - 4.0 * a * c                              # (H, W)
        distances = np.full((H, W), np.inf, dtype=np.float64)

        # CRITICAL FIX: Only evaluate where discriminant is valid AND ray direction exists (a > 0)
        # This keeps un-normalized fallback shadow rays from triggering 0/0 divisions
        valid_disc_mask = (discriminant >= 0.0) & (a > 1e-6)

        # Protect against runtime warnings by using the 'where' mask filter parameters
        sqrt_disc = np.zeros_like(discriminant)
        sqrt_disc[valid_disc_mask] = np.sqrt(discriminant[valid_disc_mask])

        # Initialize t1 and t2 with safe fallback values
        t1 = np.zeros_like(discriminant)
        t2 = np.zeros_like(discriminant)

        # Force division ONLY where valid_disc_mask is True
        np.divide((-b - sqrt_disc), (2.0 * a), out=t1, where=valid_disc_mask)
        np.divide((-b + sqrt_disc), (2.0 * a), out=t2, where=valid_disc_mask)

        t1_valid = (t1 > EPSILON) & valid_disc_mask
        t2_valid = (t2 > EPSILON) & valid_disc_mask

        inside_sphere_mask = (~t1_valid) & t2_valid     # origin inside sphere: t1 behind, t2 ahead
        front_hit_mask = t1_valid                       # standard external intersection

        distances = np.where(inside_sphere_mask, t2, distances)
        distances = np.where(front_hit_mask, t1, distances)

        return distances

    def get_normal_batch(self, hit_points: np.ndarray) -> np.ndarray:
        raw_normals = hit_points - self.center                          # (N, 3)
        magnitudes = np.linalg.norm(raw_normals, axis=-1, keepdims=True)
        magnitudes = np.where(magnitudes == 0.0, 1.0, magnitudes)
        return raw_normals / magnitudes


@dataclass
class Plane:
    point: np.ndarray       # Shape (3,)
    normal: np.ndarray      # Shape (3,)
    color: Tuple[int, int, int]
    material: Material = Material(diffuse=1.0, specular=0.0, shininess=1)

    def __post_init__(self) -> None:
        self.normal = self.normal / np.linalg.norm(self.normal)

    # --- Scalar Layer ---

    def intersect(self, ray: Ray) -> Optional[float]:
        denominator = np.dot(ray.direction, self.normal)
        if abs(denominator) < 1e-6:
            return None
        numerator = np.dot(self.point - ray.origin, self.normal)
        t = numerator / denominator
        return t if t > 1e-4 else None

    def get_normal(self, hit_point: np.ndarray) -> np.ndarray:
        return self.normal

    # --- Batch Layer ---

    def intersect_batch(self, ray_origins: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
        H, W, _ = ray_origins.shape

        denominator = np.sum(ray_directions * self.normal, axis=-1)     # (H, W)
        valid_angle_mask = np.abs(denominator) > 1e-6

        distances = np.full((H, W), np.inf, dtype=np.float64)

        # Guard denominator against zero outside the valid mask to prevent divide warning
        safe_denominator = np.where(valid_angle_mask, denominator, 1.0)
        numerator = np.sum((self.point - ray_origins) * self.normal, axis=-1)

        t = numerator / safe_denominator
        valid_hit_mask = valid_angle_mask & (t > 1e-4)

        distances = np.where(valid_hit_mask, t, distances)
        return distances

    def get_normal_batch(self, hit_points: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.normal, hit_points.shape)