import pytest
from src.vector_math import Vector
from src.geometry import Sphere  # Serving as our 3D Sphere
from src.renderer_3d import Renderer3D

def test_renderer_3d_grid_render():
    # 1. Setup a 3D scene with a red sphere sitting 5 units deep along the Z-axis
    red_sphere = Sphere(center=Vector(0.0, 0.0, 5.0), radius=1.0, color=(255, 0, 0))
    scene = [red_sphere]
    
    # 2. Initialize a 3x3 3D Renderer
    # Width = 3 pixels, Height = 3 pixels
    width = 3
    height = 3
    background_color = (0, 0, 0)
    renderer = Renderer3D(width=width, height=height, background_color=background_color)
    
    # 3. Execute a grid render looking down the Z-axis
    camera_origin = Vector(0.0, 0.0, 0.0)
    fov = 90.0
    
    image_matrix = renderer.render_3d_scene(camera_origin, fov, scene)
    
    # 4. Assert structural grid matrix dimensions
    assert len(image_matrix) == 3        # 3 Rows deep
    assert len(image_matrix[0]) == 3     # 3 Columns wide
    
    # 5. Assert geometric spatial correctness
    # Corner pixels must miss (Background)
    assert image_matrix[0][0] == (0, 0, 0)  # Top-Left
    assert image_matrix[0][2] == (0, 0, 0)  # Top-Right
    assert image_matrix[2][0] == (0, 0, 0)  # Bottom-Left
    assert image_matrix[2][2] == (0, 0, 0)  # Bottom-Right
    
    # Dead center pixel must hit (Sphere Color)
    assert image_matrix[1][1] == (255, 0, 0)  # Center