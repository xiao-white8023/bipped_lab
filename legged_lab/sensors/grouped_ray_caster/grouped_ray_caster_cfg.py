from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from .grouped_ray_caster import GroupedRayCaster


@configclass
class GroupedRayCasterCfg(MultiMeshRayCasterCfg):
    """Configuration for the GroupedRayCaster sensor."""

    class_type: type = GroupedRayCaster

    min_distance: float = 0.0
    """The minimum distance from the sensor to ray cast to. aka ignore the hits closer than this distance."""


def get_link_prim_targets(
    links: list[str],
    prefix: str = "/World/envs/env_.*/Robot/",
    suffix: str = "/visuals",
    **kwargs: dict,
) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """Build the raycast target given the list of links. It will combine and return a list of
    MultiMeshRayCasterCfg.RaycastTargetCfg.
    """
    return [MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{prefix}{link}{suffix}", **kwargs) for link in links]
