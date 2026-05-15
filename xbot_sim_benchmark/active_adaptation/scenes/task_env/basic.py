from typing import Optional

import torch

from isaaclab.assets import Articulation

from active_adaptation.envs.base import Env
import active_adaptation.envs.mdp as mdp
from active_adaptation.scenes.task_env.utils.background_3dgs_render import Background3DGSManager

class BasicEnv(Env):

    feet_name_expr: str

    def __init__(self, cfg):
        super().__init__(cfg)

        self.robot = self.scene.articulations["robot"]
        self.action_split = [act.num_joints for act in self.robot.actuators.values()]

        self.init_root_state = self.robot.data.default_root_state.clone()
        self.init_joint_pos = self.robot.data.default_joint_pos.clone()
        self.init_joint_vel = self.robot.data.default_joint_vel.clone()
        
        self.default_joint_pos = self.init_joint_pos.clone()
        
        try:
            from active_adaptation.utils.debug import DebugDraw
            self.debug_draw = DebugDraw()
            print("[INFO] Debug Draw API enabled.")
        except:
            print("[INFO] Debug Draw API disabled.")
        
        self.lookat_env_i = (
            self.scene._default_env_origins.cpu() 
            - torch.tensor(self.cfg.viewer.lookat)
        ).norm(dim=-1).argmin()
        self._gs_background = Background3DGSManager(self.cfg, self.device)

    def _reset_idx(self, env_ids: torch.Tensor):
        if not self.robot.is_fixed_base:
            init_root_state = self.command_manager.sample_init(env_ids)
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(position=default_joint_pos, velocity=default_joint_vel, env_ids=env_ids)

        self.scene.reset(env_ids)

    @torch.no_grad()
    def render_3dgs_background(
        self,
        pose_root: torch.Tensor,
        K: torch.Tensor,
        height: int,
        width: int,
    ) -> Optional[torch.Tensor]:
        if not self.is_env_initializing:
            return self._gs_background.render(pose_root, K, height, width)
        else:
            return None
