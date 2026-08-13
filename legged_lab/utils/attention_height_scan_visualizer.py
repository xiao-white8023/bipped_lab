from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class HeightScanAttentionVisualizer:
    """Viewport marker overlay for MHA attention over front height-scan points."""

    def __init__(
        self,
        env,
        policy,
        env_id: int = 0,
        top_k: int = 12,
        hot_k: int = 3,
        interval: int = 2,
        z_offset: float = 0.035,
        reachable_only: bool = True,
        reach_x_range: tuple[float, float] = (0.05, 0.65),
        reach_y_abs: float = 0.35,
        prim_path: str = "/World/Visuals/HeightScanAttention",
    ):
        self.env = env
        self.policy = policy
        self.env_id = env_id
        self.top_k = top_k
        self.hot_k = hot_k
        self.interval = max(1, interval)
        self.z_offset = z_offset
        self.reachable_only = reachable_only
        self.reach_x_range = reach_x_range
        self.reach_y_abs = reach_y_abs
        self._warned_shape = False

        marker_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "scan": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
                ),
                "focus": sim_utils.SphereCfg(
                    radius=0.026,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.05)),
                ),
                "hot": sim_utils.SphereCfg(
                    radius=0.045,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.02)),
                ),
            },
        )
        self.markers = VisualizationMarkers(marker_cfg)
        self.markers.set_visibility(True)

    def update(self, step: int):
        if step % self.interval != 0:
            return
        if not hasattr(self.policy, "get_attention_weights"):
            return

        attention = self.policy.get_attention_weights()
        if attention is None:
            return

        points_w, points_b = self._front_height_scan_points()
        if points_w is None or points_w.numel() == 0:
            self.markers.visualize(translations=torch.empty((0, 3), device=self.env.device))
            return

        weights = self._flatten_attention(attention)
        if weights is None:
            return
        if weights.numel() != points_w.shape[0]:
            if not self._warned_shape:
                print(
                    "[WARN] Attention/height-scan shape mismatch: "
                    f"attention={weights.numel()}, points={points_w.shape[0]}. "
                    "Height-scan attention visualization disabled for this model."
                )
                self._warned_shape = True
            return

        valid = torch.isfinite(points_w).all(dim=-1) & torch.isfinite(weights)
        if points_b is not None:
            valid &= torch.isfinite(points_b).all(dim=-1)
        if not valid.any():
            self.markers.visualize(translations=torch.empty((0, 3), device=self.env.device))
            return

        candidate_mask = valid
        if self.reachable_only and points_b is not None:
            candidate_mask &= self._reachable_mask(points_b)
            if not candidate_mask.any():
                candidate_mask = valid

        points = points_w[valid].clone()
        weights = weights[valid]
        candidate_indices = torch.nonzero(candidate_mask[valid], as_tuple=False).squeeze(-1)
        points[:, 2] += self.z_offset

        marker_indices = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        top_k = min(self.top_k, candidate_indices.numel())
        hot_k = min(self.hot_k, top_k)
        if top_k > 0:
            top_candidate_indices = torch.topk(weights[candidate_indices], k=top_k, largest=True).indices
            top_indices = candidate_indices[top_candidate_indices]
            marker_indices[top_indices] = 1
            if hot_k > 0:
                hot_indices = top_indices[:hot_k]
                marker_indices[hot_indices] = 2

        self.markers.visualize(translations=points, marker_indices=marker_indices)

    def _front_height_scan_points(self):
        if not hasattr(self.env, "height_scanner"):
            return None, None
        points_w = self.env.height_scanner.data.ray_hits_w[self.env_id]
        points_b = None
        if hasattr(self.env, "get_height_scan_feature_image"):
            points_b = self.env.get_height_scan_feature_image([self.env_id])[0].permute(1, 2, 0).reshape(-1, 3)
        rows = getattr(self.env, "height_scan_rows", None)
        cols = getattr(self.env, "height_scan_cols", None)
        front_col_start = getattr(self.env, "height_scan_front_col_start", 0)
        if rows is None or cols is None:
            return points_w, points_b
        points_w = points_w.reshape(rows, cols, 3)[:, front_col_start:, :].reshape(-1, 3)
        return points_w, points_b

    def _reachable_mask(self, points_b: torch.Tensor):
        x_min, x_max = self.reach_x_range
        return (
            (points_b[:, 0] >= x_min)
            & (points_b[:, 0] <= x_max)
            & (points_b[:, 1].abs() <= self.reach_y_abs)
        )

    def _flatten_attention(self, attention: torch.Tensor):
        attention = attention.detach()
        if attention.shape[0] <= self.env_id:
            return None

        weights = attention[self.env_id]
        while weights.dim() > 1:
            weights = weights.mean(dim=0)
        return weights.reshape(-1)
