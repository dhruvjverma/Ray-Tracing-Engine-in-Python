import numpy as np

from src.geometry import Sphere, Material
from src.renderer_3d import Renderer3D


def test_material_config_matte_vs_metallic():
    matte_red = Material(diffuse=1.0, specular=0.0, shininess=1)
    metallic_red = Material(diffuse=0.2, specular=0.8, shininess=64)

    light_position = (0.0, 0.0, -1.0)
    camera_origin = (0.0, 0.0, -1.0)

    sphere_matte = Sphere(
        center=np.array([0.0, 0.0, 4.0]),
        radius=1.0,
        color=(255, 0, 0),
        material=matte_red,
    )
    renderer_matte = Renderer3D(width=3, height=3, background_color=(0, 0, 0))
    matrix_matte = renderer_matte.render_3d_scene(
        camera_origin,
        60.0,
        [sphere_matte],
        light_position,
    )
    matte_pixel = matrix_matte[1][1]

    assert matte_pixel[1] == 0
    assert matte_pixel[2] == 0

    sphere_metal = Sphere(
        center=np.array([0.0, 0.0, 4.0]),
        radius=1.0,
        color=(255, 0, 0),
        material=metallic_red,
    )
    renderer_metal = Renderer3D(width=3, height=3, background_color=(0, 0, 0))
    matrix_metal = renderer_metal.render_3d_scene(
        camera_origin,
        60.0,
        [sphere_metal],
        light_position,
    )
    metal_pixel = matrix_metal[1][1]

    assert metal_pixel[1] > 0
    assert metal_pixel[2] > 0
