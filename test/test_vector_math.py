import math
import pytest

from src.vector_math import Vector

def test_vector_3d_coordinates():
    v = Vector(1.0, 2.0, 3.0)
    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 3.0

def test_vector_addition():
    v1 = Vector(1.0, 2.0, 3.0)
    v2 = Vector(4.0, 5.0, 6.0)

    result = v1 + v2
    
    assert result.x == 5.0
    assert result.y == 7.0
    assert result.z == 9.0

def test_vector_subtraction():
    v1 = Vector(5.0, 5.0, 9.0)
    v2 = Vector(1.0, 2.0, 1.0)

    result = v1 - v2
    
    assert result.x == 4.0
    assert result.y == 3.0
    assert result.z == 8.0

def test_vector_scalar_multiplication():
    v1 = Vector(1.5, -2.0, 3.0)
    scalar = 2
    
    # Test Vector * Scalar
    result_forward = v1 * scalar

    assert result_forward.x == 3.0
    assert result_forward.y == -4.0
    assert result_forward.z == 6.0

    # Test Scalar * Vector (__rmul__)
    result_reverse = scalar * v1

    assert result_reverse.x == 3.0
    assert result_reverse.y == -4.0
    assert result_reverse.z == 6.0

def test_vector_magnitude():
    v = Vector(3.0, 4.0, 0.0)

    assert v.magnitude() == 5.0

def test_vector_normalization_standard():
    v = Vector(1.0, 2.0, 3.0)

    result = v.normalize()
    
    # Hand-calculated expected values based on 1 / sqrt(14)
    expected_mag = math.sqrt(14.0)

    assert result.x == pytest.approx(1.0 / expected_mag)
    assert result.y == pytest.approx(2.0 / expected_mag)
    assert result.z == pytest.approx(3.0 / expected_mag)

    assert result.magnitude() == pytest.approx(1.0)

def test_vector_normalization_zero_vector():
    # Edge case verification: Normalizing a zero-length vector should not crash
    v = Vector(0.0, 0.0, 0.0)

    result = v.normalize()
    
    assert result.x == 0.0
    assert result.y == 0.0
    assert result.z == 0.0

def test_vector_dot_product():
    v1 = Vector(2.0, 3.0, 4.0)
    v2 = Vector(5.0, 6.0, 7.0)
    
    result = v1 @ v2
    
    assert result == 56.0