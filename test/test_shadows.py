import numpy as np

from src.geometry import Sphere
from src.renderer_3d import Renderer3D


def test_hard_shadow_occlusion():
    # Red blocker sits off the center camera ray but between the blue sphere and the light.
    blocking_sphere = Sphere(
        center=np.array([-2.0, 0.0, 5.0]),
        radius=1.0,
        color=(255, 0, 0),
    )
    shadowed_sphere = Sphere(
        center=np.array([0.0, 0.0, 8.0]),
        radius=1.0,
        color=(0, 0, 255),
    )

    scene = [blocking_sphere, shadowed_sphere]

    renderer = Renderer3D(width=3, height=3, background_color=(0, 0, 0))

    light_position = (-5.0, 0.0, 4.0)
    camera_origin = (0.0, 0.0, 0.0)

    image_matrix = renderer.render_3d_scene(
        camera_origin,
        60.0,
        scene,
        light_position,
    )

    center = image_matrix[1][1]
    assert center[0] == 0 and center[1] == 0
    assert center[2] > 0
    assert center[2] < 128
