import cupy as cp
from typing import Optional, Protocol, Tuple
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Material:
    """Configures how a surface interacts with light rays on the GPU."""
    diffuse: float = 1.0
    specular: float = 0.0
    shininess: int = 1
    reflectivity: float = 0.0  # 0.0 for completely matte, 1.0 for perfect mirror
    
    # Strictly [0, 1] normalized space for HDR.
    emission: cp.ndarray = field(
        default_factory=lambda: cp.zeros(3, dtype=cp.float32)
    )
    
    is_emissive: bool = field(init=False)

    def __post_init__(self):
        # Perform the logic cleanly using CuPy array states
        object.__setattr__(self, 'is_emissive', bool(cp.any(self.emission > 0)))

@dataclass
class Ray:
    origin: cp.ndarray      # Shape (3,) GPU array
    direction: cp.ndarray   # Shape (3,) GPU array

    def __post_init__(self) -> None:
        mag = cp.linalg.norm(self.direction)
        if mag > 0:
            self.direction = self.direction / mag

class Intersectable(Protocol):
    color: Tuple[int, int, int]
    material: Material

@dataclass
class Sphere:
    center: cp.ndarray
    radius: float
    color: Tuple[int, int, int]
    material: Material = field(default_factory=Material)

@dataclass
class Plane:
    point: cp.ndarray
    normal: cp.ndarray
    color: Tuple[int, int, int]
    material: Material = field(default_factory=lambda: Material(diffuse=1.0, specular=0.0, shininess=1))

    def __post_init__(self) -> None:
        object.__setattr__(self, 'normal', self.normal / cp.linalg.norm(self.normal))
