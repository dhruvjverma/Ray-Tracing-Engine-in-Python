import math
from src.vector_math import Vector
from src.geometry import Ray, Sphere

class Renderer3D:
    def __init__(
        self,
        width: int,
        height: int,
        background_color: tuple[int, int, int],
    ) -> None:
        self.width = width
        self.height = height
        self.background_color = background_color

    def render_3d_scene(
        self,
        camera_origin: Vector,
        fov_degrees: float,
        scene: list[Sphere],
    ) -> list[list[tuple[int, int, int]]]:
        """
        Renders a full 3D scene grid by shooting rays from the camera through a 
        2D virtual viewport window located at Z = 1.0. Returns a 2D matrix of colors.
        """
        image_matrix = []
        
        # Calculate aspect ratio to prevent image stretching
        aspect_ratio = self.width / self.height
        
        # Calculate the size scale of our virtual viewport window based on the FOV angle
        # This keeps our spatial coordinates correct relative to the camera field of view
        fov_radians = math.radians(fov_degrees)
        scale = math.tan(fov_radians / 2.0)

        # Loop through every row (Y-axis) from top to bottom
        for row in range(self.height):
            row_colors = []
            
            # Loop through every column (X-axis) from left to right
            for col in range(self.width):
                
                # 1. Map pixel indices to normalized coordinates (-1.0 to 1.0)
                # Handle single pixel edge-cases gracefully to avoid division by zero
                norm_x = 0.0 if self.width == 1 else -1.0 + (col / (self.width - 1)) * 2.0
                norm_y = 0.0 if self.height == 1 else 1.0 - (row / (self.height - 1)) * 2.0
                
                # 2. Scale coordinates by aspect ratio and field of view limits
                screen_x = norm_x * scale * aspect_ratio
                screen_y = norm_y * scale
                
                # 3. Form our ray pointing directly out into 3D world space (Looking down +Z)
                # Ray automatically normalizes direction internally thanks to __post_init__
                ray_direction = Vector(screen_x, screen_y, 1.0)
                ray = Ray(camera_origin, ray_direction)

                # 4. Perform our tracking calculations to identify the closest object hit
                closest_distance = float("inf")
                closest_color = self.background_color

                for obj in scene:
                    distance = obj.intersect(ray)

                    if distance is not None and distance < closest_distance:
                        closest_distance = distance
                        closest_color = obj.color

                row_colors.append(closest_color)
                
            image_matrix.append(row_colors)

        return image_matrix