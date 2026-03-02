from __future__ import annotations

from dataclasses import MISSING

import torch

import torch.nn.functional as F
import numpy as np
from pathlib import Path

class FieldSampler:
    """Loads and samples 3D potential fields for obstacle-aware navigation.

      - sdf ``(Nx, Ny, Nz, 1)``: signed distance field (meters).
        Positive outside obstacles, negative inside.
      - bf  ``(Nx, Ny, Nz, 3)``: boundary normal field (outward-pointing
        unit vectors derived from ``∇sdf``).
      - gf  ``(Nx, Ny, Nz, 3)``: guidance velocity field (m/s vectors
        pointing toward the goal, with obstacle-tangent blending near surfaces).

    All queries use world coordinates ``(x, y, z)``.

    Source reference: ``env_cat.py`` lines 1259-1297 (``world_to_grid``,
    ``sample_field``), ``pf_modular.py`` (field generation).
    """

    def __init__(self, path: str = MISSING, 
                 origin: tuple[float, float, float] = (-0.5, -1.0, 0.0),
                 spacing: float = 0.04, device: torch.device = torch.device('cpu')):
        """Load potential fields from a directory.

        Args:
            path: Directory containing ``sdf.npy``, ``bf.npy``, ``gf.npy``.
            origin: World coordinates ``(ox, oy, oz)`` of the grid corner
                (matches ``PFConfig.origin_w``).
            spacing: Voxel size in meters (matches ``PFConfig.voxel``,
                typically 0.04).
            device: Torch device for all tensors.
        """
        p = Path(path)
        self.sdf = torch.tensor(
            np.load(p / "sdf.npy"), dtype=torch.float32, device=device
        )[..., None]                               # (Nx, Ny, Nz, 1)
        self.bf = torch.tensor(
            np.load(p / "bf.npy"), dtype=torch.float32, device=device
        )                                          # (Nx, Ny, Nz, 3)
        self.gf = torch.tensor(
            np.load(p / "gf.npy"), dtype=torch.float32, device=device
        )                                          # (Nx, Ny, Nz, 3)

        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.spacing = spacing
        self.Nx, self.Ny, self.Nz, _ = self.sdf.shape
        self.device = device

    def _world_to_grid(self, pos: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to continuous grid indices.

        Args:
            pos: ``(P, 3)`` world positions.

        Returns:
            ``(P, 3)`` fractional grid indices.
        """
        return (pos - self.origin) / self.spacing

    def _trilinear_sample(self, field: torch.Tensor,
                          pos: torch.Tensor) -> torch.Tensor:
        """Sample a 3D voxel field via trilinear interpolation.

        Direct port of ``env_cat.py`` ``sample_field()`` from JAX to PyTorch.

        Args:
            field: ``(Nx, Ny, Nz, C)`` voxel field, C=1 for sdf, C=3 for
                bf/gf.
            pos: ``(P, 3)`` world coordinates — P is any number of query
                points (can be ``num_envs * M`` flattened).

        Returns:
            ``(P, C)`` interpolated values.
        """
        idx = self._world_to_grid(pos)              # (P, 3)
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]   # each (P,)

        x = x.clamp(0, self.Nx - 2)
        y = y.clamp(0, self.Ny - 2)
        z = z.clamp(0, self.Nz - 2)

        xi = x.floor().int()                         # (P,)
        yi = y.floor().int()
        zi = z.floor().int()
        xd = x - xi                                  # (P,) fractional part
        yd = y - yi
        zd = z - zi

        offsets = torch.tensor(
            [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
             [0,0,1],[1,0,1],[0,1,1],[1,1,1]],
            dtype=torch.int32, device=pos.device,
        )                                             # (8, 3)

        base = torch.stack([xi, yi, zi], dim=1)       # (P, 3)
        corners = base[:, None, :] + offsets[None, :, :]  # (P, 8, 3)

        # Gather corner values
        vals = field[
            corners[..., 0], corners[..., 1], corners[..., 2], :
        ]                                             # (P, 8, C)

        # Trilinear weights
        wx = torch.stack([1.0 - xd, xd], dim=1)       # (P, 2)
        wy = torch.stack([1.0 - yd, yd], dim=1)       # (P, 2)
        wz = torch.stack([1.0 - zd, zd], dim=1)       # (P, 2)

        w = (wx[:, :, None, None] * 
             wy[:, None, :, None] * 
             wz[:, None, None, :]).reshape(-1, 8)   # (P, 8)

        out = torch.einsum('pe,pec->pc', w, vals)     # (P, C)
        return out

    def sample_sdf(self, pos: torch.Tensor) -> torch.Tensor:
        """Sample the signed distance field.

        Args:
            pos: ``(P, 3)`` **or** ``(num_envs, M, 3)`` world positions.

        Returns:
            ``(P, 1)`` or ``(num_envs, M, 1)`` signed distances.
        """
        orig_shape = pos.shape
        flat = pos.reshape(-1, 3)
        out = self._trilinear_sample(self.sdf, flat)  # (P, 1)
        return out.reshape(*orig_shape[:-1], 1)

    def sample_bf(self, pos: torch.Tensor) -> torch.Tensor:
        """Sample the boundary normal field.

        Args:
            pos: ``(P, 3)`` **or** ``(num_envs, M, 3)`` world positions.

        Returns:
            ``(P, 3)`` or ``(num_envs, M, 3)`` boundary normal vectors.
        """
        orig_shape = pos.shape
        flat = pos.reshape(-1, 3)
        out = self._trilinear_sample(self.bf, flat)   # (P, 3)
        return out.reshape(*orig_shape[:-1], 3)

    def sample_gf(self, pos: torch.Tensor) -> torch.Tensor:
        """Sample the guidance velocity field.

        Args:
            pos: ``(P, 3)`` **or** ``(num_envs, M, 3)`` world positions.

        Returns:
            ``(P, 3)`` or ``(num_envs, M, 3)`` guidance velocity vectors.
        """
        orig_shape = pos.shape
        flat = pos.reshape(-1, 3)
        out = self._trilinear_sample(self.gf, flat)   # (P, 3)
        return out.reshape(*orig_shape[:-1], 3)

    def sample_all(self, pos: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample all three fields at the same positions (single grid-index
        computation, three lookups).

        Args:
            pos: ``(P, 3)`` **or** ``(num_envs, M, 3)`` world positions.

        Returns:
            Tuple ``(sdf, bf, gf)`` each shaped ``(…, C)`` matching the
            input batch dimensions.
        """
        orig_shape = pos.shape
        flat = pos.reshape(-1, 3)
        sdf_out = self._trilinear_sample(self.sdf, flat)  # (P, 1)
        bf_out  = self._trilinear_sample(self.bf, flat)    # (P, 3)
        gf_out  = self._trilinear_sample(self.gf, flat)    # (P, 3)
        return (
            sdf_out.reshape(*orig_shape[:-1], 1),
            bf_out.reshape(*orig_shape[:-1], 3),
            gf_out.reshape(*orig_shape[:-1], 3),
        )