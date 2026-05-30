import pytest
from src.vector_math import Vector
from src.geometry import Ray, Sphere

def test_ray_creation_and_auto_normalization():
    origin = Vector(0.0, 0.0, 0.0)
    direction = Vector(3.0, 0.0, 0.0)  # Length of 3
    
    ray = Ray(origin, direction)
    
    assert ray.origin == origin
    # Direction must be automatically normalized to a length of 1.0
    assert ray.direction.x == pytest.approx(1.0)
    assert ray.direction.y == pytest.approx(0.0)
    assert ray.direction.z == pytest.approx(0.0)

def test_Sphere_creation():
    center = Vector(5.0, 2.0, 0.0)
    radius = 1.5
    color = (255, 0, 0)
    
    sphere = Sphere(center, radius, color)
    
    assert sphere.center == center
    assert sphere.radius == radius
    assert sphere.color == color

def test_ray_Sphere_intersection_hit():
    ray = Ray(Vector(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    sphere = Sphere(Vector(5.0, 0.0, 0.0), radius=1.0, color=(255, 0, 0))
    
    distance = sphere.intersect(ray)
    
    # The ray should strike the front surface exactly 4.0 units away
    assert distance is not None
    assert distance == pytest.approx(4.0)

def test_ray_Sphere_intersection_miss():
    # Ray points straight up, Sphere is off to the right
    ray = Ray(Vector(0.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0))
    sphere = Sphere(Vector(5.0, 0.0, 0.0), radius=1.0, color=(255, 0, 0))
    
    distance = sphere.intersect(ray)
    
    assert distance is None

def test_ray_Sphere_intersection_object_behind_camera():
    # Sphere is to the left (-5, 0), ray points to the right (+1, 0)
    ray = Ray(Vector(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    sphere = Sphere(Vector(-5.0, 0.0, 0.0), radius=1.0, color=(255, 0, 0))
    
    assert sphere.intersect(ray) is None

def test_ray_Sphere_intersection_camera_inside_Sphere():
    # Camera is inside a giant Sphere of radius 5.0
    ray = Ray(Vector(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0))
    sphere = Sphere(Vector(0.0, 0.0, 0.0), radius=5.0, color=(255, 0, 0))
    
    distance = sphere.intersect(ray)
    assert distance is not None
    assert distance == pytest.approx(5.0)