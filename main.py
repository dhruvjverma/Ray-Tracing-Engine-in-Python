import time
import cupy as cp
import numpy as np

from PIL import Image
from src.geometry import Sphere, Material, Plane
from src.renderer_3d import Renderer3D

def main():
    WIDTH = 800
    HEIGHT = 600
    BACKGROUND_COLOR = (30, 32, 40)  

    print(f"Initializing CuPy context for RTX 4070 at {WIDTH}x{HEIGHT}...")
    renderer = Renderer3D(width=WIDTH, height=HEIGHT, background_color=BACKGROUND_COLOR)

    camera_origin = (0.0, 0.0, -1.0)
    field_of_view = 65.0

    matte_clay_red = Material(diffuse=0.95, specular=0.0, shininess=1, reflectivity=0.0)
    glossy_plastic_green = Material(diffuse=0.85, specular=0.4, shininess=32, reflectivity=0.15)
    polished_mirror_blue = Material(diffuse=0.2, specular=0.9, shininess=128, reflectivity=0.8)
    floor_mat = Material(diffuse=0.9, specular=0.2, shininess=16, reflectivity=0.25)
    overhead_light_mat = Material(diffuse=0.0, specular=0.0, shininess=1, reflectivity=0.0, emission=cp.array([1.0, 1.0, 1.0], dtype=cp.float32))

    scene = [
        Sphere(center=cp.array([0.0, 4.0, 3.8], dtype=cp.float32), radius=1.2, color=(255, 255, 255), material=overhead_light_mat),
        Sphere(center=cp.array([-2.2, 0.0, 4.5], dtype=cp.float32), radius=0.9, color=(255, 45, 45), material=matte_clay_red),
        Sphere(center=cp.array([0.0, 0.0, 4.5], dtype=cp.float32), radius=0.9, color=(45, 225, 45), material=glossy_plastic_green),
        Sphere(center=cp.array([2.2, 0.0, 4.5], dtype=cp.float32), radius=0.9, color=(45, 95, 255), material=polished_mirror_blue),
        Plane(point=cp.array([0.0, -0.9, 0.0], dtype=cp.float32), normal=cp.array([0.0, 1.0, 0.0], dtype=cp.float32), color=(200, 200, 200), material=floor_mat)
    ]

    MAX_BOUNCE_DEPTH = 4    
    AA_SAMPLES = 2          # 2x AA turns into a 4x SSAA Grid filter
    SAMPLES_PER_PIXEL = 512 

    print("-" * 64)
    print("Executing Vectorized GPU Engine Pipeline (CuPy Backend):")
    print(" - Rendering Style:  Unbiased Monte Carlo Path Tracer")
    print(f" - Anti-Aliasing:   {AA_SAMPLES * AA_SAMPLES}x SSAA Grid Filtering")
    print(f" - Total Ray Paths:  {SAMPLES_PER_PIXEL} SPP (Random Jittered)")
    print(f" - Trace Bounce Limit: {MAX_BOUNCE_DEPTH} Recursive Depth")
    print("-" * 64)

    start_time = time.perf_counter()

    image_matrix = renderer.render_3d_scene(
        camera_origin_tuple=camera_origin, 
        fov_degrees=field_of_view, 
        scene=scene, 
        max_depth=MAX_BOUNCE_DEPTH,
        aa_samples=AA_SAMPLES,
        spp=SAMPLES_PER_PIXEL
    )
    
    elapsed_seconds = time.perf_counter() - start_time
    
    print("\nSaving host image array matrix...")
    img = Image.fromarray(image_matrix, mode="RGB")
    img.save("render_output.png")
    
    print(f"⏱️ Total Time Taken: {int(elapsed_seconds // 60)}m {elapsed_seconds % 60:.2f}s")
    print("Success! Created 'render_output.png' via GPU acceleration.")

if __name__ == "__main__":
    main()
