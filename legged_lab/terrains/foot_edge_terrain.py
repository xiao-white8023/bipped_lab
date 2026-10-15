"""Task-local terrain importer for the paper's virtual edge cylinders.

The physical terrain is imported unchanged. Only a separate mesh copy is used
by the sharp-edge detector. No changes to the installed IsaacLab are needed.
"""

from __future__ import annotations

from dataclasses import fields
import copy

import trimesh

from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from legged_lab.perception.foot_edge_geometry import EdgeGeometry, EdgeGeometryCfg


class FootEdgeTerrainImporter(TerrainImporter):
    """Retain the generated world mesh and build one shared edge query."""

    def __init__(self, cfg):
        self._foot_edge_imported_meshes = []
        self.foot_edge_geometry = None
        self.foot_edge_mesh = None
        super().__init__(cfg)
        if self._foot_edge_imported_meshes:
            self.foot_edge_mesh = (
                self._foot_edge_imported_meshes[0]
                if len(self._foot_edge_imported_meshes) == 1
                else trimesh.util.concatenate(self._foot_edge_imported_meshes)
            )
            self.foot_edge_geometry = EdgeGeometry(
                self.foot_edge_mesh, cfg.geometry_cfg, self.device
            )
        elif cfg.terrain_type == "plane":
            # Infinite flat ground has no terrain feature edges.
            import numpy as np

            self.foot_edge_geometry = EdgeGeometry.from_edges(
                np.empty((0, 2, 3), dtype=np.float32), cfg.geometry_cfg, self.device
            )
        else:
            raise ValueError(
                "Foot edge reward currently supports generated mesh or plane terrain; "
                "USD-only terrain must first provide its world-space triangle mesh."
            )
        self._foot_edge_imported_meshes.clear()

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        # Keep the original argument untouched, including its vertices/faces.
        self._foot_edge_imported_meshes.append(mesh.copy())
        super().import_mesh(name, mesh)


@configclass
class FootEdgeTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = FootEdgeTerrainImporter
    geometry_cfg: EdgeGeometryCfg = EdgeGeometryCfg()


def with_foot_edge_geometry(terrain_cfg, geometry_cfg: EdgeGeometryCfg):
    """Copy only the task's terrain configuration, preserving all physics fields."""
    values = {
        item.name: copy.deepcopy(getattr(terrain_cfg, item.name))
        for item in fields(TerrainImporterCfg)
        if item.init and item.name != "class_type"
    }
    return FootEdgeTerrainImporterCfg(
        **values, geometry_cfg=copy.deepcopy(geometry_cfg)
    )
