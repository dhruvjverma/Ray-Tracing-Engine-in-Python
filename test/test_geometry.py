import numpy as np
import pytest

from src.geometry import Ray, Sphere

def test_ray_creation_and_auto_normalization():
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([3.0, 0.0, 0.0])  # Length of 3

    ray = Ray(origin, direction)

    np.testing.assert_allclose(ray.origin, origin)
    np.testing.assert_allclose(
        ray.direction,
        np.array([1.0, 0.0, 0.0]),
        rtol=1e-7,
        atol=1e-7,
    )

def test_sphere_creation():
    center = np.array([5.0, 2.0, 0.0])
    radius = 1.5
    color = (255, 0, 0)

    sphere = Sphere(center, radius, color)

    np.testing.assert_allclose(sphere.center, center)
    assert sphere.radius == radius
    assert sphere.color == color

def test_ray_sphere_intersection_hit():
    ray = Ray(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )

    sphere = Sphere(
        np.array([5.0, 0.0, 0.0]),
        radius=1.0,
        color=(255, 0, 0),
    )

    distance = sphere.intersect(ray)

    assert distance is not None
    assert distance == pytest.approx(4.0)

def test_ray_sphere_intersection_miss():
    ray = Ray(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )

    sphere = Sphere(
        np.array([5.0, 0.0, 0.0]),
        radius=1.0,
        color=(255, 0, 0),
    )

    assert sphere.intersect(ray) is None

def test_ray_sphere_intersection_object_behind_camera():
    ray = Ray(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )

    sphere = Sphere(
        np.array([-5.0, 0.0, 0.0]),
        radius=1.0,
        color=(255, 0, 0),
    )

    assert sphere.intersect(ray) is None

def test_ray_sphere_intersection_camera_inside_sphere():
    ray = Ray(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )

    sphere = Sphere(
        np.array([0.0, 0.0, 0.0]),
        radius=5.0,
        color=(255, 0, 0),
    )

    distance = sphere.intersect(ray)

    assert distance is not None
    assert distance == pytest.approx(5.0)