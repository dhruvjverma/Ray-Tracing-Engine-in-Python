import math
import time
import cupy as cp
import numpy as np

from typing import List, Tuple
from dataclasses import dataclass
from src.geometry import Intersectable, Sphere, Plane

def srgb_to_linear(srgb_array: cp.ndarray) -> cp.ndarray:
    """Converts sRGB [0,1] to linear light using CUDA-accelerated element-wise ops."""
    return cp.where(srgb_array <= 0.04045,
                    srgb_array / 12.92,
                    ((srgb_array + 0.055) / 1.055) ** 2.4)

@dataclass
class SceneBuffers:
    """Linearized Struct-of-Arrays scene representation fully pinned on the GPU."""
    is_sphere: cp.ndarray
    centers: cp.ndarray
    radii_sq: cp.ndarray
    plane_normals: cp.ndarray
    colors_linear: cp.ndarray
    diffuse: cp.ndarray
    reflectivity: cp.ndarray
    emission: cp.ndarray
    is_emissive: cp.ndarray

class Renderer3D:
    def __init__(self, width: int, height: int, background_color: Tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.MAX_DIST = 1e30 
        
        # Initialize background color directly as a GPU tensor
        bg_cpu = np.array(background_color, dtype=np.float32) / 255.0
        self.background_linear = srgb_to_linear(cp.asarray(bg_cpu))
        
        # CuPy Random Generator handles multi-threaded CUDA-native parallel RNG generation
        self.rng = cp.random.default_rng(seed=42)

    def _build_scene_buffers(self, scene: List[Intersectable]) -> SceneBuffers:
        """Flattens Python geometry objects directly into unified CuPy GPU allocations."""
        N = len(scene)
        buf = SceneBuffers(
            is_sphere=cp.zeros(N, dtype=bool),
            centers=cp.zeros((N, 3), dtype=cp.float32),
            radii_sq=cp.zeros(N, dtype=cp.float32),
            plane_normals=cp.zeros((N, 3), dtype=cp.float32),
            colors_linear=cp.zeros((N, 3), dtype=cp.float32),
            diffuse=cp.zeros(N, dtype=cp.float32),
            reflectivity=cp.zeros(N, dtype=cp.float32),
            emission=cp.zeros((N, 3), dtype=cp.float32),
            is_emissive=cp.zeros(N, dtype=bool)
        )
        
        for i, obj in enumerate(scene):
            color_gpu = cp.asarray(np.array(obj.color, dtype=np.float32) / 255.0)
            buf.colors_linear[i] = srgb_to_linear(color_gpu)
            buf.diffuse[i] = obj.material.diffuse
            buf.reflectivity[i] = obj.material.reflectivity
            buf.emission[i] = cp.asarray(obj.material.emission)
            buf.is_emissive[i] = obj.material.is_emissive

            if isinstance(obj, Sphere):
                buf.is_sphere[i] = True
                buf.centers[i] = cp.asarray(obj.center)
                buf.radii_sq[i] = obj.radius ** 2
            elif isinstance(obj, Plane):
                buf.is_sphere[i] = False
                buf.centers[i] = cp.asarray(obj.point)
                buf.plane_normals[i] = cp.asarray(obj.normal)
                
        return buf

    def render_3d_scene(
        self,
        camera_origin_tuple: Tuple[float, float, float],
        fov_degrees: float,
        scene: List[Intersectable],
        max_depth: int = 4,
        aa_samples: int = 1,
        spp: int = 128,
    ) -> np.ndarray:
        
        camera_origin = cp.asarray(np.array(camera_origin_tuple, dtype=np.float32))
        ss_width, ss_height = self.width * aa_samples, self.height * aa_samples
        
        # Unified tracking allocations on VRAM
        total_accumulated_color = cp.zeros((ss_height, ss_width, 3), dtype=cp.float32)
        scene_buf = self._build_scene_buffers(scene)
        
        aspect_ratio = cp.float32(self.width / self.height)
        scale = cp.float32(math.tan(math.radians(fov_degrees) / 2.0))
        cols, rows = cp.meshgrid(cp.arange(ss_width, dtype=cp.float32), cp.arange(ss_height, dtype=cp.float32))

        print(f"Beginning CUDA-accelerated path tracing sequence ({spp} SPP)...")
        start_time = time.perf_counter()

        for sample in range(spp):
            if sample % max(1, spp // 20) == 0 or sample == spp - 1:
                elapsed = time.perf_counter() - start_time
                eta = (elapsed / max(sample, 1)) * (spp - sample)
                print(f"\r  [{100 * sample / spp:5.1f}%] Sample {sample}/{spp} | VRAM Tracked | ETA: {eta:.1f}s", end="", flush=True)

            # Primary Ray Jitter (Fully calculated on GPU cores)
            jitter_x = self.rng.random((ss_height, ss_width), dtype=cp.float32) - 0.5
            jitter_y = self.rng.random((ss_height, ss_width), dtype=cp.float32) - 0.5
            norm_x = -1.0 + ((cols + 0.5 + jitter_x) / ss_width) * 2.0
            norm_y =  1.0 - ((rows + 0.5 + jitter_y) / ss_height) * 2.0

            dirs = cp.stack([norm_x * scale * aspect_ratio, norm_y * scale, cp.ones_like(norm_x)], axis=-1)
            magnitudes = cp.sqrt(dirs[..., 0]**2 + dirs[..., 1]**2 + dirs[..., 2]**2)[..., cp.newaxis]
            ray_directions = dirs / magnitudes
            ray_origins = cp.broadcast_to(camera_origin, ray_directions.shape).copy()

            sample_color = cp.zeros((ss_height, ss_width, 3), dtype=cp.float32)
            throughput = cp.ones((ss_height, ss_width, 3), dtype=cp.float32)
            active_rays_mask = cp.ones((ss_height, ss_width), dtype=bool)

            for depth in range(max_depth):
                # .any() creates a tiny CPU synchronizer block, but it is necessary for loop safety
                if not cp.any(active_rays_mask):
                    break

                # --- Vectorized CUDA Batch Intersection Kernel ---
                O = ray_origins[..., cp.newaxis, :]
                D = ray_directions[..., cp.newaxis, :]
                
                # Spheres Math
                of = O - scene_buf.centers
                b = 2.0 * cp.sum(of * D, axis=-1)
                c = cp.sum(of * of, axis=-1) - scene_buf.radii_sq
                disc = b**2 - 4.0 * c
                valid_sphere = (disc >= 0) & scene_buf.is_sphere
                
                sqrt_disc = cp.where(valid_sphere, cp.sqrt(cp.maximum(0, disc)), 0.0)
                t1 = cp.where(valid_sphere, (-b - sqrt_disc) / 2.0, self.MAX_DIST)
                t2 = cp.where(valid_sphere, (-b + sqrt_disc) / 2.0, self.MAX_DIST)
                t_spheres = cp.minimum(cp.where(t1 > 1e-4, t1, self.MAX_DIST), cp.where(t2 > 1e-4, t2, self.MAX_DIST))
                
                # Planes Math
                denom = cp.sum(D * scene_buf.plane_normals, axis=-1)
                valid_plane = (cp.abs(denom) > 1e-6) & (~scene_buf.is_sphere)
                safe_denom = cp.where(valid_plane, denom, 1.0)
                num = cp.sum((scene_buf.centers - O) * scene_buf.plane_normals, axis=-1)
                t_planes = cp.where(valid_plane & ((num / safe_denom) > 1e-4), num / safe_denom, self.MAX_DIST)
                
                # Selection
                t_all = cp.where(scene_buf.is_sphere, t_spheres, t_planes)
                closest_idx = cp.argmin(t_all, axis=-1)
                closest_distances = cp.min(t_all, axis=-1)
                
                hit_mask = closest_distances < self.MAX_DIST
                current_hit_mask = active_rays_mask & hit_mask
                background_mask = active_rays_mask & (~hit_mask)

                # Background Evaluation
                if cp.any(background_mask):
                    sample_color[background_mask] += throughput[background_mask] * self.background_linear
                    active_rays_mask[background_mask] = False

                if not cp.any(current_hit_mask):
                    break

                # Geometry and Normal Extraction
                hit_idx = closest_idx[current_hit_mask]
                ray_origins[current_hit_mask] += ray_directions[current_hit_mask] * closest_distances[current_hit_mask, np.newaxis]
                hit_points = ray_origins[current_hit_mask]
                
                is_sphere_hit = scene_buf.is_sphere[hit_idx][..., cp.newaxis]
                sphere_normals = hit_points - scene_buf.centers[hit_idx]
                s_mags = cp.sqrt(sphere_normals[..., 0]**2 + sphere_normals[..., 1]**2 + sphere_normals[..., 2]**2)[..., cp.newaxis]
                sphere_normals = cp.where(s_mags > 1e-6, sphere_normals / s_mags, 0.0)
                normals = cp.zeros_like(ray_directions)
                normals[current_hit_mask] = cp.where(is_sphere_hit, sphere_normals, scene_buf.plane_normals[hit_idx])

                # Batch VRAM Property Lookups
                mat_emission = scene_buf.emission[hit_idx]
                mat_is_emissive = scene_buf.is_emissive[hit_idx]

                # Terminate emissive paths
                sample_color[current_hit_mask] += throughput[current_hit_mask] * mat_emission
                terminate_on_light = cp.zeros_like(active_rays_mask)
                terminate_on_light[current_hit_mask] = mat_is_emissive
                active_rays_mask &= ~terminate_on_light
                
                current_hit_mask = active_rays_mask & hit_mask
                if not cp.any(current_hit_mask):
                    break

                # --- Material Routing Kernel ---
                ray_dirs = ray_directions[current_hit_mask]
                hit_norms = normals[current_hit_mask]
                
                dot_i_n = cp.sum(ray_dirs * hit_norms, axis=-1, keepdims=True)
                reflected_dirs = ray_dirs - 2.0 * dot_i_n * hit_norms
                
                rand_vecs = self.rng.standard_normal(hit_norms.shape, dtype=cp.float32)
                r_mags = cp.sqrt(rand_vecs[..., 0]**2 + rand_vecs[..., 1]**2 + rand_vecs[..., 2]**2)[..., cp.newaxis]
                rand_vecs = cp.where(r_mags < 1e-12, 1.0, rand_vecs / r_mags)
                dots = cp.sum(rand_vecs * hit_norms, axis=-1, keepdims=True)
                diffuse_dirs = cp.where(dots < 0.0, -rand_vecs, rand_vecs)

                random_rolls = self.rng.random(ray_dirs.shape[0], dtype=cp.float32)[..., cp.newaxis]
                
                subset_idx = closest_idx[current_hit_mask]
                sub_color = scene_buf.colors_linear[subset_idx]
                sub_refl = scene_buf.reflectivity[subset_idx][..., cp.newaxis]
                sub_diff = scene_buf.diffuse[subset_idx][..., np.newaxis]

                use_mirror = random_rolls < sub_refl
                ray_directions[current_hit_mask] = cp.where(use_mirror, reflected_dirs, diffuse_dirs)

                # Unbiased Probability Adjustment Weights
                mirror_tp = sub_color / cp.clip(sub_refl, 1e-6, 1.0)
                diffuse_tp = sub_color * sub_diff / cp.clip(1.0 - sub_refl, 1e-6, 1.0)
                throughput[current_hit_mask] *= cp.where(use_mirror, mirror_tp, diffuse_tp)

                ray_origins[current_hit_mask] += normals[current_hit_mask] * 1e-4

                # Russian Roulette Path Termination
                if depth >= 2:
                    lum = 0.299 * throughput[..., 0] + 0.587 * throughput[..., 1] + 0.114 * throughput[..., 2]
                    survival_prob = cp.clip(lum, 0.1, 0.95)
                    rr_rolls = self.rng.random((ss_height, ss_width), dtype=cp.float32)
                    
                    terminate_mask = active_rays_mask & (rr_rolls > survival_prob)
                    active_rays_mask &= ~terminate_mask
                    throughput[active_rays_mask] /= survival_prob[active_rays_mask, cp.newaxis]

            sample_color = cp.where(cp.isfinite(sample_color), sample_color, 0.0)
            total_accumulated_color += sample_color

        print("\nCUDA Trace loop complete.")
        
        # HDR Tone Mapping -> Gamma Correction Pipeline on GPU
        averaged_linear = total_accumulated_color / cp.float32(spp)
        tone_mapped = averaged_linear / (1.0 + averaged_linear)

        if aa_samples > 1:
            reshaped = tone_mapped.reshape(self.height, aa_samples, self.width, aa_samples, 3)
            tone_mapped = reshaped.mean(axis=(1, 3))

        gamma_corrected = cp.power(cp.clip(tone_mapped, 0.0, 1.0), 1.0 / 2.2)
        final_rgb = (gamma_corrected * 255.0).astype(cp.uint8)
        
        # Explicit transfer from GPU memory (CuPy Array) to Host memory (NumPy Array)
        return cp.asnumpy(final_rgb)
