import numpy as np
from PIL import Image
from src.geometry import Sphere, Material, Plane
from src.renderer_3d import Renderer3D

def main():
    WIDTH = 800
    HEIGHT = 600
    BACKGROUND_COLOR = (25, 25, 35)

    print(f"Initializing batch NumPy execution context at {WIDTH}x{HEIGHT}...")
    renderer = Renderer3D(width=WIDTH, height=HEIGHT, background_color=BACKGROUND_COLOR)

    camera_origin = (0.0, 0.0, -1.0)
    light_position = (-3.0, 4.0, 1.0)
    field_of_view = 65.0

    matte_clay = Material(diffuse=1.0, specular=0.0, shininess=1)
    glossy_plastic = Material(diffuse=0.7, specular=0.4, shininess=32)
    polished_metallic = Material(diffuse=0.2, specular=0.8, shininess=128)
    floor_mat = Material(diffuse=0.8, specular=0.1, shininess=10)

    # Injecting optimized NumPy tracking vectors cleanly across the scene graph
    scene = [
        Sphere(center=np.array([-2.2, 0.0, 4.5], dtype=np.float64), radius=0.9, color=(255, 65, 65), material=matte_clay),
        Sphere(center=np.array([0.0, 0.0, 4.5], dtype=np.float64), radius=0.9, color=(65, 225, 65), material=glossy_plastic),
        Sphere(center=np.array([2.2, 0.0, 4.5], dtype=np.float64), radius=0.9, color=(65, 105, 255), material=polished_metallic),
        Plane(point=np.array([0.0, -0.9, 0.0], dtype=np.float64), normal=np.array([0.0, 1.0, 0.0], dtype=np.float64), color=(180, 180, 180), material=floor_mat)
    ]

    print("Running loopless-per-pixel scene trace matrix operations...")
    image_matrix = renderer.render_3d_scene(camera_origin, field_of_view, scene, light_position)
    print("Matrix parsing complete. Writing flat pixel stream to image...")

    # Flatten buffer array and convert directly into standard output canvases
    flat_pixel_buffer = [tuple(pixel) for row in image_matrix for pixel in row]

    img = Image.new("RGB", (WIDTH, HEIGHT))
    img.putdata(flat_pixel_buffer)
    img.save("render_output.png")
    print("Success! Created vectorized 'render_output.png'.")

if __name__ == "__main__":
    main()