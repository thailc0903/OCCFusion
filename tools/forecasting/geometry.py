"""Physical full SE(3) and discrete XYZ semantic pull resampling, NumPy only."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tools import evaluate_occupancy as occupancy


def se3(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('SE(3) must be finite 4x4')
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-10) or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-8):
        raise ValueError('Invalid rigid SE(3) rotation/bottom row')
    return matrix


def transform_points(points, target_from_source):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 metric points')
    matrix = se3(target_from_source)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def anchor_from_source(global_from_anchor, global_from_source):
    return se3(np.linalg.inv(se3(global_from_anchor)) @ se3(global_from_source))


def centres(indices, bounds=occupancy.BOUNDS, voxel_size=occupancy.VOXEL_SIZE):
    return np.asarray(bounds[:3], dtype=np.float64) + (np.asarray(indices, dtype=np.float64) + .5) * np.asarray(voxel_size)


@dataclass(frozen=True)
class WarpedLabels:
    labels: np.ndarray
    in_bounds: np.ndarray
    known: np.ndarray

    @property
    def valid(self):
        return self.in_bounds & self.known


def pull_labels(source, anchor_from_source_matrix, *, bounds=occupancy.BOUNDS,
                voxel_size=occupancy.VOXEL_SIZE, grid=occupancy.GRID, chunk_size=100_000):
    """Cell lookup at anchor centres. Unknown/OOB remain 255 with distinct masks."""
    source = np.asarray(source)
    grid = tuple(grid)
    minimum, maximum = np.asarray(bounds[:3], dtype=np.float64), np.asarray(bounds[3:], dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    if source.shape != grid or source.dtype.kind not in 'iu' or np.any(~((source <= 16) | (source == 255))) or np.any(source < 0):
        raise ValueError('Invalid source XYZ semantic grid')
    if size.shape != (3,) or np.any(size <= 0) or not np.allclose(minimum + size * grid, maximum, rtol=0, atol=1e-10) or chunk_size < 1:
        raise ValueError('Grid/bounds/voxel size mismatch')
    source_from_anchor = np.linalg.inv(se3(anchor_from_source_matrix))
    output = np.full(source.size, 255, dtype=np.uint8)
    in_bounds = np.zeros(source.size, dtype=bool)
    known = np.zeros(source.size, dtype=bool)
    for start in range(0, source.size, chunk_size):
        flat = np.arange(start, min(start + chunk_size, source.size))
        indices = np.column_stack(np.unravel_index(flat, grid, order='C'))
        points = transform_points(centres(indices, bounds, size), source_from_anchor)
        inside = np.all((points >= minimum) & (points < maximum), axis=1)
        lookup = np.floor((points[inside] - minimum) / size).astype(np.int64)
        values = source[tuple(lookup.T)]
        output[flat[inside]] = values
        in_bounds[flat] = inside
        known[flat[inside]] = values != 255
    return WarpedLabels(output.reshape(grid), in_bounds.reshape(grid), known.reshape(grid))
