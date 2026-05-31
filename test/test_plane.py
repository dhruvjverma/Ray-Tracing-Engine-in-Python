import pytest

from src.vector_math import Vector
from src.geometry import Ray, Plane

def test_plane_intersection_looking_down():
    # Floor plane at Y = -1.0, pointing straight up
    floor = Plane(point=Vector(0.0, -1.0, 0.0), normal=Vector(0.0, 1.0, 0.0), color=(100, 100, 100))
    
    # Camera at origin shooting a ray slightly downward and forward
    ray = Ray(origin=Vector(0.0, 0.0, 0.0), direction=Vector(0.0, -1.0, 1.0))
    
    distance = floor.intersect(ray)
    
    assert distance is not None
    # With a 45-degree downward angle, it should hit the floor exactly at sqrt(2) units away
    assert pytest.approx(distance, abs=1e-4) == 1.41421356