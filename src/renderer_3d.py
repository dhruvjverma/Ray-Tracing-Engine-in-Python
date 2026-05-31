import math
import numpy as np
from typing import List, Tuple
from src.geometry import Intersectable, Ray


def _to_linear_batch(colors: np.ndarray) -> np.ndarray:
    """Converts a whole image block from sRGB space to linear space."""
    return ((colors / 255.0) ** 2.2) * 255.0


class Renderer3D:
    def __init__(
        self,
        width: int,
        height: int,
        background_color: Tuple[int, int, int],
    ) -> None:
        self.width = width
        self.height = height
        self.background_color = background_color

    def render_3d_scene(
        self,
        camera_origin_tuple: Tuple[float, float, float],
        fov_degrees: float,
        scene: List[Intersectable],
        light_position_tuple: Tuple[float, float, float],
    ) -> List[List[Tuple[int, int, int]]]:
        """Entry method coordinating the vectorized multi-stage data pipelines."""
        camera_origin = np.array(camera_origin_tuple, dtype=np.float64)
        light_position = np.array(light_position_tuple, dtype=np.float64)

        # Helper 1: Setup global grid coordinates and ray directions
        ray_origins, ray_directions = self._build_primary_rays(camera_origin, fov_degrees)

        # Helper 2: Handle ray intersections across every primitive object in the scene
        closest_obj_indices, closest_distances, hit_mask = self._intersect_scene(
            ray_origins, ray_directions, scene
        )

        # Create output channels filled with the default background color configuration
        out_channels = np.zeros((self.height, self.width, 3), dtype=np.float64)
        out_channels[:, :] = self.background_color

        # Short-circuit if nothing is intersected by the view frustum
        if not np.any(hit_mask):
            return out_channels.astype(np.uint8).tolist()

        # Isolate and extract exact geometric world space coordinate hit locations
        hit_points = ray_origins + ray_directions * closest_distances[..., np.newaxis]

        # Helper 3: Pull surface normals polymorphically using masked blocks
        normals = self._compute_normals(hit_points, closest_obj_indices, hit_mask, scene)

        # Helper 4: Evaluate shadow visibility using a complete secondary intersection loop
        shadow_mask = self._compute_shadows(hit_points, normals, hit_mask, light_position, scene)

        # Helper 5: Perform Phong illumination shading, clamp intensities, and run gamma correction
        lit_colors = self._shade(
            hit_points, ray_directions, normals, shadow_mask, hit_mask, closest_obj_indices, light_position, scene
        )

        # Splice the processed lighting pixel data back on top of the default background canvas layers
        out_channels[hit_mask] = lit_colors[hit_mask]

        # Cast to standard output structure array matrices safely
        return out_channels.astype(np.uint8).tolist()

    def _build_primary_rays(self, camera_origin: np.ndarray, fov_degrees: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generates coordinate meshes for screen pixels instantly using np.meshgrid."""
        aspect_ratio = self.width / self.height
        scale = math.tan(math.radians(fov_degrees) / 2.0)

        cols = np.arange(self.width, dtype=np.float64)
        rows = np.arange(self.height, dtype=np.float64)

        col_grid, row_grid = np.meshgrid(cols, rows)

        norm_x = 0.0 if self.width == 1 else -1.0 + (col_grid / (self.width - 1)) * 2.0
        norm_y = 0.0 if self.height == 1 else 1.0 - (row_grid / (self.height - 1)) * 2.0

        # Construct direction tensors
        dirs = np.stack([norm_x * scale * aspect_ratio, norm_y * scale, np.ones_like(norm_x)], axis=-1)
        magnitudes = np.linalg.norm(dirs, axis=-1, keepdims=True)
        ray_directions = dirs / magnitudes

        # Broadcast static origin out to match full array matrix spatial properties
        ray_origins = np.broadcast_to(camera_origin, ray_directions.shape)

        return ray_origins, ray_directions

    def _intersect_scene(
        self, ray_origins: np.ndarray, ray_directions: np.ndarray, scene: List[Intersectable]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds a 3D distance buffer and gathers closest intersection data across objects."""
        # Create a distance tracking layer stacked along axis 0: Shape (num_objects, H, W)
        all_distances = np.stack([obj.intersect_batch(ray_origins, ray_directions) for obj in scene], axis=0)

        closest_obj_indices = np.argmin(all_distances, axis=0)
        closest_distances = np.min(all_distances, axis=0)

        # Background pixels reside where even the absolute minimum closest distance reads as infinite
        hit_mask = closest_distances < np.inf

        return closest_obj_indices, closest_distances, hit_mask

    def _compute_normals(
        self, hit_points: np.ndarray, closest_obj_indices: np.ndarray, hit_mask: np.ndarray, scene: List[Intersectable]
    ) -> np.ndarray:
        """Extracts localized normals polymorphically using target object masks."""
        normals = np.zeros_like(hit_points)

        for i, obj in enumerate(scene):
            obj_mask = hit_mask & (closest_obj_indices == i)
            if np.any(obj_mask):
                # Symmetric extraction pass using the newly introduced batch endpoints
                normals[obj_mask] = obj.get_normal_batch(hit_points[obj_mask])

        return normals

    def _compute_shadows(
        self, hit_points: np.ndarray, normals: np.ndarray, hit_mask: np.ndarray, light_position: np.ndarray, scene: List[Intersectable]
    ) -> np.ndarray:
        """Traces secondary shadow occlusion structures across the scene graph."""
        H, W, _ = hit_points.shape
        EPSILON = 1e-4

        # Isolate active foreground regions before establishing secondary tracing structures
        shadow_mask = np.zeros((H, W), dtype=bool)

        if not np.any(hit_mask):
            return shadow_mask

        # Offset locations along normal lines to isolate primary objects from precision self-occlusion acne
        shadow_origins = np.zeros_like(hit_points)
        shadow_origins[hit_mask] = hit_points[hit_mask] + normals[hit_mask] * EPSILON

        shadow_dirs = np.zeros_like(hit_points)
        shadow_dirs[hit_mask] = light_position - shadow_origins[hit_mask]

        light_distances = np.zeros((H, W), dtype=np.float64)
        light_distances[hit_mask] = np.linalg.norm(shadow_dirs[hit_mask], axis=-1)

        # Run secondary ray traces across every object in the scene
        shadow_distances = np.stack([obj.intersect_batch(shadow_origins, shadow_dirs) for obj in scene], axis=0)

        # Evaluate occlusion status inside primary foreground masks
        shadow_mask[hit_mask] = np.any(shadow_distances[:, hit_mask] < light_distances[hit_mask], axis=0)

        return shadow_mask

    def _shade(
        self,
        hit_points: np.ndarray,
        ray_directions: np.ndarray,
        normals: np.ndarray,
        shadow_mask: np.ndarray,
        hit_mask: np.ndarray,
        closest_obj_indices: np.ndarray,
        light_position: np.ndarray,
        scene: List[Intersectable],
    ) -> np.ndarray:
        """Executes parallel Phong illumination transformations, clamping, and gamma encoding."""
        H, W, _ = hit_points.shape
        AMBIENT_INTENSITY = 0.1
        GAMMA_EXPONENT = 1.0 / 2.2

        # Initialize linear output color buffers
        r_linear = np.zeros((H, W), dtype=np.float64)
        g_linear = np.zeros((H, W), dtype=np.float64)
        b_linear = np.zeros((H, W), dtype=np.float64)

        # Separate unmasked calculations by iterating over the unique objects present in the scene
        for i, obj in enumerate(scene):
            obj_mask = hit_mask & (closest_obj_indices == i)
            if not np.any(obj_mask):
                continue

            # Linearize primitive source sRGB colors before illumination math operations occur
            color_r = _to_linear_batch(obj.color[0])
            color_g = _to_linear_batch(obj.color[1])
            color_b = _to_linear_batch(obj.color[2])
            mat = obj.material

            # Define localized masks for shadowed vs illuminated regions
            shd_submask = obj_mask & shadow_mask
            lit_submask = obj_mask & ~shadow_mask

            # Path A: Shading for elements covered by shadowed regions
            if np.any(shd_submask):
                r_linear[shd_submask] = color_r * AMBIENT_INTENSITY * mat.diffuse
                g_linear[shd_submask] = color_g * AMBIENT_INTENSITY * mat.diffuse
                b_linear[shd_submask] = color_b * AMBIENT_INTENSITY * mat.diffuse

            # Path B: Full ambient, diffuse, and specular illumination tracking paths
            if np.any(lit_submask):
                to_light = light_position - hit_points[lit_submask]
                light_dist = np.linalg.norm(to_light, axis=-1, keepdims=True)
                # Eliminate possible zero division warnings inside inactive allocations
                light_dist = np.where(light_dist == 0.0, 1.0, light_dist)
                light_dir = to_light / light_dist

                # Calculate diffuse factors using clean np.sum syntax
                n_dot_l = np.sum(normals[lit_submask] * light_dir, axis=-1)
                diffuse_intensity = np.maximum(AMBIENT_INTENSITY, n_dot_l) * mat.diffuse

                # Calculate specular reflections
                reflection_vec = (normals[lit_submask] * (2.0 * n_dot_l)[..., np.newaxis]) - light_dir
                view_vec = -ray_directions[lit_submask]
                r_dot_v = np.maximum(0.0, np.sum(reflection_vec * view_vec, axis=-1))
                specular_intensity = (r_dot_v ** mat.shininess) * mat.specular

                # Accumulate distinct streams into global floating point linear arrays
                r_linear[lit_submask] = (color_r * diffuse_intensity) + (255.0 * specular_intensity)
                g_linear[lit_submask] = (color_g * diffuse_intensity) + (255.0 * specular_intensity)
                b_linear[lit_submask] = (color_b * diffuse_intensity) + (255.0 * specular_intensity)

        # Enforce boundary restrictions inside linear space fields prior to gamma transformation
        r_linear = np.clip(r_linear, 0.0, 255.0)
        g_linear = np.clip(g_linear, 0.0, 255.0)
        b_linear = np.clip(b_linear, 0.0, 255.0)

        # Execute display monitor sRGB gamma curve adjustments in parallel
        r_out = (((r_linear / 255.0) ** GAMMA_EXPONENT) * 255.0).astype(np.int32)
        g_out = (((g_linear / 255.0) ** GAMMA_EXPONENT) * 255.0).astype(np.int32)
        b_out = (((b_linear / 255.0) ** GAMMA_EXPONENT) * 255.0).astype(np.int32)

        return np.stack([r_out, g_out, b_out], axis=-1)