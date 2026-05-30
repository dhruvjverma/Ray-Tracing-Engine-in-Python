from PIL import Image

from src.vector_math import Vector
from src.geometry import Sphere
from src.renderer_3d import Renderer3D

def main():
    # 1. Define Image Resolution and Output Configurations
    # I'll use a 800x600 resolution (4:3 Aspect Ratio)
    WIDTH = 800
    HEIGHT = 600
    BACKGROUND_COLOR = (30, 30, 40)  # A nice dark, atmospheric twilight blue

    print(f"Initializing engine canvas... Resolving space at {WIDTH} x {HEIGHT}.")
    renderer = Renderer3D(width=WIDTH, height=HEIGHT, background_color=BACKGROUND_COLOR)

    # 2. Configure our 3D Camera coordinates
    camera_origin = Vector(0.0, 0.0, 0.0)  # Placed at the world origin
    field_of_view = 60.0

    # 3. Populate the 3D World Scene
    # I'll place a red sphere right down the positive Z-axis
    red_sphere = Sphere(
        center=Vector(0.0, 0.0, 4.0),       # 4 units deep directly ahead of the camera
        radius=1,                         # Wide enough to fill a good portion of the screen
        color=(255, 65, 65)                 # Red
    )

    # Add a second object slightly off to the side to test depth occlusion
    blue_sphere = Sphere(
        center=Vector(2.0, 2.0, 6.0),       # Placed further down the right side of the screen
        radius=0.5,
        color=(65, 105, 225)                # Blue
    )

    scene = [red_sphere, blue_sphere]

    # 4. Invoke the 3D Render Engine to trace the rays
    print("Tracing rays into 3D world space... Please wait...")
    image_matrix = renderer.render_3d_scene(camera_origin, field_of_view, scene)
    print("Ray tracing complete! Processing image grid arrays.")

    # 5. Flatten the 2D matrix data for Pillow
    # Pillow's `putdata` method expects a long 1D sequence of RGB tuples:
    # [(r,g,b), (r,g,b), (r,g,b), ...] read row-by-row, left-to-right.
    flat_pixel_buffer = []
    for row in image_matrix:
        for pixel_color in row:
            flat_pixel_buffer.append(pixel_color)

    # 6. Hand the pixel buffer off to a Pillow canvas object
    # "RGB" creates a standard 3-channel color image canvas
    img = Image.new("RGB", (WIDTH, HEIGHT))
    img.putdata(flat_pixel_buffer)

    # 7. Write the canvas object out to a PNG image file
    output_filename = "render_output.png"
    img.save(output_filename)
    print(f"Success! Image drawn and saved cleanly as '{output_filename}'.")

if __name__ == "__main__":
    main()