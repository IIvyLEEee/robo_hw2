"""
Dual-arm placement with per-object bbox sampling and radius-based separation.
"""
import random
from collections.abc import Mapping, Sequence


class DualArmPlacementStrategy:
    def __init__(self, params: dict, context: dict):
        self.positions_relative = True
        self.clearance = float(params.get("clearance", 0.05))
        self.max_sample_tries = int(params.get("max_sample_tries", 100))
        self.max_plan_tries = int(params.get("max_plan_tries", 50))
        self.num_top_objects = int(params.get("num_top_objects", 2))

        self.objs_id_map = context["objs_id_map"]
        self.top_objs = context["top_objs"]
        self.btm_objs = context["btm_objs"]

        if not self.btm_objs:
            raise ValueError("Dual-arm placement requires at least one bottom object.")
        if len(self.top_objs) < self.num_top_objects:
            raise ValueError(
                f"Dual-arm placement requires at least {self.num_top_objects} top objects, "
                f"got {len(self.top_objs)}."
            )

    def _parse_bbox(self, cfg: dict) -> tuple[float, float, float, float]:
        bbox = cfg.get("bbox", None)
        if bbox is None:
            raise ValueError(f"Object '{cfg.get('name', '<unnamed>')}' is missing bbox config.")

        if isinstance(bbox, Mapping):
            x_range = bbox.get("x_range")
            y_range = bbox.get("y_range")
            if x_range is None or y_range is None:
                raise ValueError(f"Invalid bbox dict for '{cfg.get('name', '<unnamed>')}'.")
            return (float(x_range[0]), float(x_range[1]), float(y_range[0]), float(y_range[1]))

        if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 2:
            (x0, y0), (x1, y1) = bbox
            return (float(x0), float(x1), float(y0), float(y1))

        raise ValueError(f"Unsupported bbox format for '{cfg.get('name', '<unnamed>')}'.")

    def _sample_position(self, bounds, radius, placed):
        x_min, x_max, y_min, y_max = bounds
        if x_max <= x_min or y_max <= y_min:
            return None

        for _ in range(self.max_sample_tries):
            # In dual-arm placement, bbox is the valid center sampling region.
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)

            ok = True
            for px, py, pr in placed:
                if (x - px) ** 2 + (y - py) ** 2 < (radius + pr + self.clearance) ** 2:
                    ok = False
                    break
            if ok:
                return x, y
        return None

    def _place_object(self, obj_idx, placed):
        cfg = self.objs_id_map[obj_idx]
        bounds = self._parse_bbox(cfg)
        radius = float(cfg.get("size", 0.05))
        pos_xy = self._sample_position(bounds, radius, placed)
        if pos_xy is None:
            return None

        z_offset = float(cfg.get("z_offset", 0.0))
        placed.append((pos_xy[0], pos_xy[1], radius))
        return (pos_xy[0], pos_xy[1], z_offset)

    def plan(self, env_ids):
        plan = []
        for env_id in env_ids.tolist():
            placements = []
            placed = []
            bottom_idx = None
            selected_top_ids = None

            for _ in range(self.max_plan_tries):
                placements.clear()
                placed.clear()

                bottom_idx = random.choice(self.btm_objs)
                top_candidates = [idx for idx in self.top_objs if idx != bottom_idx]
                if len(top_candidates) < self.num_top_objects:
                    continue
                selected_top_ids = random.sample(top_candidates, self.num_top_objects)

                bottom_pos = self._place_object(bottom_idx, placed)
                if bottom_pos is None:
                    continue
                placements.append({"obj_id": bottom_idx, "pos": bottom_pos})

                top_positions = []
                ok = True
                for top_idx in selected_top_ids:
                    top_pos = self._place_object(top_idx, placed)
                    if top_pos is None:
                        ok = False
                        break
                    top_positions.append((top_idx, top_pos))
                if not ok:
                    continue

                for top_idx, top_pos in top_positions:
                    placements.append({"obj_id": top_idx, "pos": top_pos})
                break

            if not placements or bottom_idx is None or not selected_top_ids:
                raise RuntimeError("Failed to sample dual-arm placements with bbox/radius constraints.")

            target_t = selected_top_ids[0]
            plan.append(
                {
                    "env_id": env_id,
                    "target_b": bottom_idx,
                    "target_t": target_t,
                    "placements": placements,
                    "relative": True,
                }
            )

        return plan
