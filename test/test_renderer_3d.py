import numpy as np

from src.geometry import Sphere
from src.renderer_3d import Renderer3D


def test_renderer_3d_grid_render():
    red_sphere = Sphere(
        center=np.array([0.0, 0.0, 5.0]),
        radius=1.0,
        color=(255, 0, 0),
    )
    scene = [red_sphere]

    width = 3
    height = 3
    background_color = (0, 0, 0)
    renderer = Renderer3D(width=width, height=height, background_color=background_color)

    camera_origin = (0.0, 0.0, 0.0)
    light_position = (0.0, 0.0, -1.0)
    fov = 90.0

    image_matrix = renderer.render_3d_scene(camera_origin, fov, scene, light_position)

    assert len(image_matrix) == 3
    assert len(image_matrix[0]) == 3

    assert image_matrix[0][0] == [0, 0, 0]
    assert image_matrix[0][2] == [0, 0, 0]
    assert image_matrix[2][0] == [0, 0, 0]
    assert image_matrix[2][2] == [0, 0, 0]

    assert image_matrix[1][1] == [255, 0, 0]
