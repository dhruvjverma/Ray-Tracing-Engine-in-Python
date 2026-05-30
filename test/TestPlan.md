# TDD Implementation Plan: 2D/3D Ray Tracer in Python

This document tracks the Test-Driven Development (TDD) blueprint for building a static image ray tracer from scratch. Phases 1–4 are implemented under `test/`; Phase 5 remains planned for future work.

**Test layout**

| File | Module under test |
|------|-------------------|
| `test/test_vector_math.py` | `src/vector_math.py` |
| `test/test_geometry.py` | `src/geometry.py` |
| `test/test_renderer.py` | `src/renderer.py` |
| `test/test_renderer_3d.py` | `src/renderer_3d.py` |

Run all tests: `pytest test/`

---

## Phase 1: Vector Math — **Done**

Implemented in `test/test_vector_math.py`. The `Vector` class is 3D from the start (`x`, `y`, `z`).

| Test | What it verifies |
|------|------------------|
| `test_vector_3d_coordinates` | Construction stores `x`, `y`, `z` |
| `test_vector_addition` | `__add__` component-wise |
| `test_vector_subtraction` | `__sub__` component-wise |
| `test_vector_scalar_multiplication` | `__mul__` and `__rmul__` (vector × scalar and scalar × vector) |
| `test_vector_magnitude` | `magnitude()` via Pythagorean theorem (e.g. `(3, 4, 0)` → `5.0`) |
| `test_vector_dot_product` | Dot product via `@` operator |
| `test_vector_normalization_standard` | `normalize()` yields unit length |
| `test_vector_normalization_zero_vector` | Zero vector normalizes safely without crashing |

---

## Phase 2: Ray and Circle Geometry — **Done**

Implemented in `test/test_geometry.py`.

### 2.1 Ray representation

| Test | What it verifies |
|------|------------------|
| `test_ray_creation_and_auto_normalization` | `Ray` stores `origin` and auto-normalizes `direction` to length 1 |

### 2.2 Circle definition

| Test | What it verifies |
|------|------------------|
| `test_circle_creation` | `Circle` stores `center`, `radius`, and `color` (RGB tuple) |

### 2.3 Intersection (hit)

| Test | What it verifies |
|------|------------------|
| `test_ray_circle_intersection_hit` | Ray from `(0,0,0)` along `+X` hits circle at `(5,0,0)` with radius `1` → distance `4.0` |

**Scenario:** Camera at origin, direction `(1, 0, 0)`, circle center `(5, 0, 0)`, radius `1` → hit at distance `4.0`.

### 2.4 Miss and edge cases

| Test | What it verifies |
|------|------------------|
| `test_ray_circle_intersection_miss` | Ray aimed away (e.g. `+Y`) returns `None` |
| `test_ray_circle_intersection_object_behind_camera` | Circle behind the ray (`-X`) returns `None` |
| `test_ray_circle_intersection_camera_inside_circle` | Ray from inside a large circle still returns a valid forward distance |

---

## Phase 3: 1D Render Loop (2D Engine) — **Done**

Implemented in `test/test_renderer.py` via `Renderer2D`.

### 3.1 1D viewport sweeper

| Test | What it verifies |
|------|------------------|
| `test_renderer_1d_pixel_sweep` | Three custom ray directions → `[Black, Red, Black]` for a red circle ahead |

**Setup:** Circle at `(5, 0, 0)`, resolution 3, directions angled up / straight / angled down; center pixel hits.

### 3.2 Additional renderer behavior (beyond original plan)

| Test | What it verifies |
|------|------------------|
| `test_renderer_closest_object_occlusion` | When multiple circles lie on the same ray, the closest hit wins (scene order independent) |
| `test_renderer_fov_direction_generation` | `generate_fov_directions(90°)` produces three normalized directions spanning −45° to +45° |

---

## Phase 4: 3D Space — **Done**

### 4.1 3D vectors

Covered in Phase 1 (`test_vector_math.py`). Geometry and renderer tests use `z = 0` or full 3D coordinates; 2D scenarios remain valid in the XY plane.

### 4.2 2D viewport grid (rows and columns)

Implemented in `test/test_renderer_3d.py` via `Renderer3D`. `Circle` acts as a sphere (same intersection math in 3D).

| Test | What it verifies |
|------|------------------|
| `test_renderer_3d_grid_render` | 3×3 image matrix: corners miss (background), center pixel hits red sphere at `(0, 0, 5)` |

**Setup:** Camera at origin, 90° FOV, sphere center `(0, 0, 5)`, radius `1`.

---

## Phase 5: Future Improvements

Not yet implemented. Planned extensions:

### 5.1 Diffuse shading

* **Test (Red):** Assert that a ray striking a sphere dead-center (directly facing a light source) evaluates to full color brightness, whereas a ray striking the curved flank of a sphere evaluates to a darker shade.
* **Implementation (Green):** When an intersection is confirmed, calculate the surface normal vector at that coordinate. Compute the dot product between this normal and the vector pointing toward the light source to scale color intensity.

### 5.2 Shadow verification

* **Test (Red):** Design a test scene featuring two spheres lined up perfectly in front of a light source. Run an intersection test on the *further* sphere. Assert that its output color matches the ambient shadow color (dark), not the illuminated color.
* **Implementation (Green):** At the point of intersection, fire a secondary "Shadow Ray" directed toward the light source coordinate. If this ray encounters any geometric intersections prior to reaching the light source, truncate the diffuse calculation and apply shadow shading.
