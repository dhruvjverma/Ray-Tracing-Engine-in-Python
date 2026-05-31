import pytest
from src.vector_math import Vector
from src.geometry import Sphere, Material
from src.renderer_3d import Renderer3D

def test_material_config_matte_vs_metallic():
    # 1. Define distinct material settings
    matte_red = Material(diffuse=1.0, specular=0.0, shininess=1)
    metallic_red = Material(diffuse=0.2, specular=0.8, shininess=64)
    
    # 2. Place a light directly over the camera shoulder
    light_position = Vector(0.0, 0.0, -1.0)
    
    # 3. Render the Matte Sphere dead center
    sphere_matte = Sphere(center=Vector(0.0, 0.0, 4.0), radius=1.0, color=(255, 0, 0), material=matte_red)
    renderer_matte = Renderer3D(width=1, height=1, background_color=(0, 0, 0))
    matrix_matte = renderer_matte.render_3d_scene(
        camera_origin=Vector(0.0, 0.0, -1.0),
        fov_degrees=60.0,
        scene=[sphere_matte],
        light_position=light_position
    )
    matte_pixel = matrix_matte[0][0]
    
    # Verify Matte object strictly has no specular bleeding
    assert matte_pixel[1] == 0  # Green channel is perfectly zero
    assert matte_pixel[2] == 0  # Blue channel is perfectly zero
    
    # 4. Render the Metallic Sphere dead center
    sphere_metal = Sphere(center=Vector(0.0, 0.0, 4.0), radius=1.0, color=(255, 0, 0), material=metallic_red)
    renderer_metal = Renderer3D(width=1, height=1, background_color=(0, 0, 0))
    matrix_metal = renderer_metal.render_3d_scene(
        camera_origin=Vector(0.0, 0.0, -1.0),
        fov_degrees=60.0,
        scene=[sphere_metal],
        light_position=light_position
    )
    metal_pixel = matrix_metal[0][0]
    
    # Verify Metallic object contains strong specular highlights
    assert metal_pixel[1] > 0   # White specular light is bleeding into the green channel
    assert metal_pixel[2] > 0   # White specular light is bleeding into the blue channel