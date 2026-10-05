"""Thin selection-replay agent for Stage-4 official PDMS evaluation.

Instead of running the Drive-JEPA model, this agent returns a precomputed
candidate trajectory per token (one row of ``proposals`` selected offline by
``pdm_score - lambda * r_hat``). The selection file is an ``.npz`` written by
``scripts/experience/export_selections.py`` with:

  tokens       (S,)   str   scene tokens (matching scene_metadata.initial_token)
  trajectories (S,8,3) float32 local-frame poses of the selected candidate

Used with ``run_pdm_score`` exactly like a normal agent: the scene loader and
metric cache are shared, only ``compute_trajectory`` is replayed.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class ReplaySelectionAgent(AbstractAgent):
    """Replays per-token precomputed trajectories from an npz selection file."""

    def __init__(self, selection_file: str):
        super().__init__(requires_scene=True)
        self._selection_file = selection_file
        sel = np.load(Path(selection_file), allow_pickle=False)
        self._tokens = [str(t) for t in sel["tokens"]]
        self._trajs = sel["trajectories"].astype(np.float32)
        self._idx = {t: i for i, t in enumerate(self._tokens)}

    def name(self) -> str:
        return f"replay_selection({Path(self._selection_file).stem})"

    def get_sensor_config(self) -> SensorConfig:
        # no sensors needed -- selection is precomputed
        return SensorConfig.build_all_sensors(False)

    def initialize(self) -> None:
        pass

    def compute_trajectory(
        self, current_input: AgentInput, scene: Optional[Scene] = None
    ) -> Trajectory:
        token = scene.scene_metadata.initial_token
        return Trajectory(poses=self._trajs[self._idx[token]])
