import numpy as np
import pytest

from src.geometry import Ray, Plane


def test_plane_intersection_looking_down():
    floor = Plane(
        point=np.array([0.0, -1.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        color=(100, 100, 100),
    )

    ray = Ray(
        origin=np.array([0.0, 0.0, 0.0]),
        direction=np.array([0.0, -1.0, 1.0]),
    )

    distance = floor.intersect(ray)

    assert distance is not None
    assert pytest.approx(distance, abs=1e-4) == 1.41421356
