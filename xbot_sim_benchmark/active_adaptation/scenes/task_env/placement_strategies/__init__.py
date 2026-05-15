"""Placement strategies retained for the xbot benchmark."""
from .dual_arm_placement import DualArmPlacementStrategy

__all__ = [
    "DualArmPlacementStrategy",
    "create_placement_strategy",
]


def create_placement_strategy(name: str, params: dict, context: dict):
    strategy_map = {
        "dual_arm": DualArmPlacementStrategy,
        "dual": DualArmPlacementStrategy,
    }
    if name not in strategy_map:
        raise ValueError(f"Unsupported placement strategy: {name}")
    return strategy_map[name](params, context)
