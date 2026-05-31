import pytest

from src.vector_math import Vector
from src.geometry import Sphere
from src.renderer_3d import Renderer3D

def test_hard_shadow_occlusion():
    # 1. Setup a scene where 1 sphere completely blocks the light from the 2nd sphere.

    # Front Sphere
    blocking_sphere = Sphere(center = Vector(0.0, 0.0, 4.0), radius = 1.0, color = (255, 0, 0))
    # Rear Sphere
    shadowed_sphere = Sphere(center = Vector(0.0, 0.0, 8.0), radius = 1.0, color = (0, 0, 255))

    scene = [blocking_sphere, shadowed_sphere]

    # 2. Set up a single-pixel renderer targeting the dead-center ray
    renderer = Renderer3D(width = 1, height = 1, background_color = (0, 0, 0))

    # 3. Defining light position (placed on the left side)
    light_position = Vector(-5.0, 0.0, 4.0)

    # 4. Execute the renderer
    image_matrix = renderer.render_3d_scene(
        camera_origin = Vector(0.0, 0.0, 0.0),
        fov_degrees=60.0,
        scene = scene,
        light_position = light_position
    )

    # 5. Assertion
    """
    The camera ray goes straight through (0,0,0) -> (0,0,1). It hits the blue sphere's back layer.
    Because the red sphere blocks the light path from that hit point to the light, 
    the pixel should NOT be bright blue (0, 0, 255). It should be shadowed (black or dark ambient).
    """
    assert image_matrix[0][0] == (0, 0, 0)