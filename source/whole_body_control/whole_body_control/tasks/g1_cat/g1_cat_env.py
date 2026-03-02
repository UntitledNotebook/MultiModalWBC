from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.sensors import FrameTransformer

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
    yaw_quat,
)

from .g1_cat_env_cfg import G1CatEnvCfg
from .utils import FieldSampler
from .constants import (
    ACTION_JOINT_NAMES, OBS_JOINT_NAMES, ALL_JOINT_NAMES,
    KPS, KDS, TORQUE_LIMIT, NUM_FIELD_SITES,
)

@dataclass
class CommandState:
    """Velocity commands and their history."""

    command: torch.Tensor  # [move_flag, x_vel, y_vel, yaw_vel], shape: (num_envs, 4)
    last_command: torch.Tensor  # Previous timestep's command, shape: (num_envs, 4)
    command_delay: torch.Tensor  # Delayed velocity command (simulates sensor latency), shape: (num_envs, 4)

@dataclass
class ActionState:
    """Action history for smoothness rewards and motor control."""

    motor_targets: torch.Tensor  # Target joint positions for PD control, shape: (num_envs, 29)
    last_act: torch.Tensor  # Previous action applied, shape: (num_envs, 12)
    last_last_act: torch.Tensor  # Action from two timesteps ago, shape: (num_envs, 12)

@dataclass
class VelocityState:
    """Linear and angular velocities in various frames."""

    local_lin_vel: torch.Tensor  # Linear velocity in pelvis local frame, shape: (num_envs, 3)
    global_lin_vel: torch.Tensor  # Linear velocity in world frame, shape: (num_envs, 3)
    global_ang_vel: torch.Tensor  # Angular velocity in world frame, shape: (num_envs, 3)
    last_joint_vel: torch.Tensor  # Previous joint velocities, shape: (num_envs, 29)
    last_feet_vel: torch.Tensor  # Previous feet vertical velocities, shape: (num_envs, 2)

@dataclass
class NaviFrameState:
    """Navigation frame transforms and projected states.

    The navi frame is a gravity-aligned frame where x points in the robot's heading direction.
    """

    navi2world_rot: torch.Tensor  # Rotation matrix from navi to world frame, shape: (num_envs, 3, 3)
    navi2world_pose: torch.Tensor  # 4x4 pose matrix from navi to world frame, shape: (num_envs, 4, 4)
    navi_torso_rpy: torch.Tensor  # Torso roll-pitch-yaw in navi frame, shape: (num_envs, 3)
    navi_torso_lin_vel: torch.Tensor  # Torso linear velocity in navi frame, shape: (num_envs, 3)
    navi_torso_ang_vel: torch.Tensor  # Torso angular velocity in navi frame, shape: (num_envs, 3)
    navi_pelvis_rpy: torch.Tensor  # Pelvis roll-pitch-yaw in navi frame, shape: (num_envs, 3)
    navi_pelvis_lin_vel: torch.Tensor  # Pelvis linear velocity in navi frame, shape: (num_envs, 3)
    navi_pelvis_ang_vel: torch.Tensor  # Pelvis angular velocity in navi frame, shape: (num_envs, 3)

@dataclass
class GaitState:
    """Gait phase control for bipedal locomotion.

    Uses a sinusoidal oscillator to generate gait patterns for left/right legs.
    All fields are batched: first dimension = num_envs.
    """

    stop_timestep: torch.Tensor   # Steps remaining in stop transition, shape: (num_envs,)
    phase: torch.Tensor           # Gait phase [left, right] (rad), shape: (num_envs, 2)
    phase_dt: torch.Tensor        # Phase increment per ctrl step (rad), shape: (num_envs,)
    gait_mask: torch.Tensor       # Gait mask -1/0/1 (swing/trans/stance), shape: (num_envs, 2)
    gait_freq: torch.Tensor       # Gait frequency (Hz), shape: (num_envs,)
    foot_height: torch.Tensor     # Target foot swing height (m), shape: (num_envs,)


@dataclass
class DomainRandState:
    """Domain randomization parameters for PD control.

    These parameters are randomly sampled each episode to improve sim-to-real transfer.
    All fields are batched: first dimension = num_envs.
    """

    kp_scale: torch.Tensor        # Proportional gain scale, shape: (num_envs,)
    kd_scale: torch.Tensor        # Derivative gain scale, shape: (num_envs,)
    rfi_lim_scale: torch.Tensor   # RFI noise limit × torque_limit, shape: (num_envs, 29)

@dataclass
class PushState:
    """External perturbation state for robustness training.

    Random pushes are applied to the robot base at random intervals.
    All fields are batched: first dimension = num_envs.
    """

    push: torch.Tensor                # Push force [x, y], shape: (num_envs, 2)
    push_step: torch.Tensor           # Steps since last push, shape: (num_envs,)
    push_interval_steps: torch.Tensor # Interval between pushes (steps), shape: (num_envs,)

@dataclass
class FieldState:
    """Potential field data stored compactly as (num_envs, 11, C) tensors.

    Body-site layout along dim-1 (mirrors all_poses in env_cat.py step()):
        0     : head          (torso head site)
        1     : pelv          (imu_in_pelvis)
        2     : tors          (imu_in_torso)
        3-4   : feet          (left_foot, right_foot)
        5-6   : hands         (left_palm, right_palm)
        7-8   : knees         (left_knee, right_knee)
        9-10  : shlds         (left_shoulder, right_shoulder)

    GF = Guidance Field (goal-directed velocity, shape C=3)
    BF = Boundary Field (outward obstacle normal, shape C=3)
    DF = Signed Distance Field (metres, shape C=1)
    """

    # Current fields
    gf: torch.Tensor        # (num_envs, 11, 3) – guidance field
    bf: torch.Tensor        # (num_envs, 11, 3) – boundary field
    df: torch.Tensor        # (num_envs, 11, 1) – signed-distance field

    # Delayed fields (updated every delay_update_interval steps)
    gf_delay: torch.Tensor  # (num_envs, 11, 3)
    bf_delay: torch.Tensor  # (num_envs, 11, 3)
    df_delay: torch.Tensor  # (num_envs, 11, 1)

    # ------------------------------------------------------------------
    # Named properties — current
    # ------------------------------------------------------------------

    @property
    def headgf(self) -> torch.Tensor: return self.gf[:, 0:1]   # (N, 1, 3)
    @property
    def headbf(self) -> torch.Tensor: return self.bf[:, 0:1]
    @property
    def headdf(self) -> torch.Tensor: return self.df[:, 0:1]   # (N, 1, 1)

    @property
    def pelvgf(self) -> torch.Tensor: return self.gf[:, 1:2]
    @property
    def pelvbf(self) -> torch.Tensor: return self.bf[:, 1:2]
    @property
    def pelvdf(self) -> torch.Tensor: return self.df[:, 1:2]

    @property
    def torsgf(self) -> torch.Tensor: return self.gf[:, 2:3]
    @property
    def torsbf(self) -> torch.Tensor: return self.bf[:, 2:3]
    @property
    def torsdf(self) -> torch.Tensor: return self.df[:, 2:3]

    @property
    def feetgf(self) -> torch.Tensor: return self.gf[:, 3:5]   # (N, 2, 3)
    @property
    def feetbf(self) -> torch.Tensor: return self.bf[:, 3:5]
    @property
    def feetdf(self) -> torch.Tensor: return self.df[:, 3:5]   # (N, 2, 1)

    @property
    def handsgf(self) -> torch.Tensor: return self.gf[:, 5:7]
    @property
    def handsbf(self) -> torch.Tensor: return self.bf[:, 5:7]
    @property
    def handsdf(self) -> torch.Tensor: return self.df[:, 5:7]

    @property
    def kneesgf(self) -> torch.Tensor: return self.gf[:, 7:9]
    @property
    def kneesbf(self) -> torch.Tensor: return self.bf[:, 7:9]
    @property
    def kneesdf(self) -> torch.Tensor: return self.df[:, 7:9]

    @property
    def shldsgf(self) -> torch.Tensor: return self.gf[:, 9:11]
    @property
    def shldsbf(self) -> torch.Tensor: return self.bf[:, 9:11]
    @property
    def shldsdf(self) -> torch.Tensor: return self.df[:, 9:11]

    # ------------------------------------------------------------------
    # Named properties — delayed
    # ------------------------------------------------------------------

    @property
    def headgf_delay(self) -> torch.Tensor: return self.gf_delay[:, 0:1]
    @property
    def headbf_delay(self) -> torch.Tensor: return self.bf_delay[:, 0:1]
    @property
    def headdf_delay(self) -> torch.Tensor: return self.df_delay[:, 0:1]

    @property
    def pelvgf_delay(self) -> torch.Tensor: return self.gf_delay[:, 1:2]
    @property
    def pelvbf_delay(self) -> torch.Tensor: return self.bf_delay[:, 1:2]
    @property
    def pelvdf_delay(self) -> torch.Tensor: return self.df_delay[:, 1:2]

    @property
    def torsgf_delay(self) -> torch.Tensor: return self.gf_delay[:, 2:3]
    @property
    def torsbf_delay(self) -> torch.Tensor: return self.bf_delay[:, 2:3]
    @property
    def torsdf_delay(self) -> torch.Tensor: return self.df_delay[:, 2:3]

    @property
    def feetgf_delay(self) -> torch.Tensor: return self.gf_delay[:, 3:5]
    @property
    def feetbf_delay(self) -> torch.Tensor: return self.bf_delay[:, 3:5]
    @property
    def feetdf_delay(self) -> torch.Tensor: return self.df_delay[:, 3:5]

    @property
    def handsgf_delay(self) -> torch.Tensor: return self.gf_delay[:, 5:7]
    @property
    def handsbf_delay(self) -> torch.Tensor: return self.bf_delay[:, 5:7]
    @property
    def handsdf_delay(self) -> torch.Tensor: return self.df_delay[:, 5:7]

    @property
    def kneesgf_delay(self) -> torch.Tensor: return self.gf_delay[:, 7:9]
    @property
    def kneesbf_delay(self) -> torch.Tensor: return self.bf_delay[:, 7:9]
    @property
    def kneesdf_delay(self) -> torch.Tensor: return self.df_delay[:, 7:9]

    @property
    def shldsgf_delay(self) -> torch.Tensor: return self.gf_delay[:, 9:11]
    @property
    def shldsbf_delay(self) -> torch.Tensor: return self.bf_delay[:, 9:11]
    @property
    def shldsdf_delay(self) -> torch.Tensor: return self.df_delay[:, 9:11]


@dataclass
class BodyState:
    """Body part positions and velocities in world frame.

    Also includes delayed odometry for sensor simulation.
    """

    head_pos: torch.Tensor  # Head site position in world frame, shape: (num_envs, 3)
    head_vel: torch.Tensor  # Head linear velocity, shape: (num_envs, 3)
    pelv_pos: torch.Tensor  # Pelvis site position, shape: (num_envs, 3)
    tors_pos: torch.Tensor  # Torso site position, shape: (num_envs, 3)
    feet_pos: torch.Tensor  # Feet site positions, shape: (num_envs, 2, 3)
    feet_vel: torch.Tensor  # Feet linear velocities, shape: (num_envs, 2, 3)
    hands_pos: torch.Tensor  # Hand site positions, shape: (num_envs, 2, 3)
    hands_vel: torch.Tensor  # Hand linear velocities, shape: (num_envs, 2, 3)
    knees_pos: torch.Tensor  # Knee site positions, shape: (num_envs, 2, 3)
    shlds_pos: torch.Tensor  # Shoulder site positions, shape: (num_envs, 2, 3)
    odom_delay: torch.Tensor  # Delayed odometry [x, y, z, qw, qx, qy, qz], shape: (num_envs, 7)

class G1CatEnv(DirectRLEnv):
    cfg: G1CatEnvCfg
    def __init__(self, cfg: G1CatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # ------------------------------------------------------------------ joints
        self._action_joint_ids, _ = self.robot.find_joints(ACTION_JOINT_NAMES)
        self._obs_joint_ids, _ = self.robot.find_joints(OBS_JOINT_NAMES)
        self._all_joint_ids, _ = self.robot.find_joints(ALL_JOINT_NAMES)

        # ------------------------------------------------------------------ PD
        self._kps = torch.tensor(KPS, dtype=torch.float32, device=self.device)           # (29,)
        self._kds = torch.tensor(KDS, dtype=torch.float32, device=self.device)           # (29,)
        self._torque_limit = torch.tensor(TORQUE_LIMIT, dtype=torch.float32, device=self.device)  # (29,)

        # ------------------------------------------------------------------ potential fields
        self.field_sampler = FieldSampler(
            path=cfg.pf_config.path,
            origin=cfg.pf_config.origin,
            spacing=cfg.pf_config.dx,
            device=self.device,
        )
        self._ctrl_dt = cfg.decimation * cfg.sim.dt  # 0.02 s

        # ------------------------------------------------------------------ default state
        # default_joint_pos shape: (num_envs, 29) — take [0] if broadcast needed
        self._default_joint_pos = self.robot.data.default_joint_pos[0].clone()  # (29,)

        # ------------------------------------------------------------------ soft joint limits
        # Shape: (num_envs, num_joints, 2) -> take [0] since all envs have same limits
        self._soft_lowers = self.robot.data.soft_joint_pos_limits[0, :, 0]  # (29,)
        self._soft_uppers = self.robot.data.soft_joint_pos_limits[0, :, 1]  # (29,)

        # ------------------------------------------------------------------ frame transformer
        self.frame_transformer: FrameTransformer = self.scene.sensors["frame_transformer"]
        # Resolve indices of each site group within target_frame_names
        head_idx,  _ = self.frame_transformer.find_bodies("head")
        pelv_idx,  _ = self.frame_transformer.find_bodies("imu_in_pelvis")
        tors_idx,  _ = self.frame_transformer.find_bodies("imu_in_torso")
        feet_idx,  _ = self.frame_transformer.find_bodies(["left_foot", "right_foot"], preserve_order=True)
        hands_idx, _ = self.frame_transformer.find_bodies(["left_palm", "right_palm"], preserve_order=True)
        knees_idx, _ = self.frame_transformer.find_bodies(["left_knee", "right_knee"], preserve_order=True)
        shlds_idx, _ = self.frame_transformer.find_bodies(["left_shoulder", "right_shoulder"], preserve_order=True)

        # Convenience per-group indices (used outside _reset_idx too)
        self._head_frame_idx  = head_idx[0]
        self._pelv_frame_idx  = pelv_idx[0]
        self._tors_frame_idx  = tors_idx[0]
        self._feet_frame_idx  = feet_idx   # [left, right]
        self._hands_frame_idx = hands_idx  # [left, right]
        self._knees_frame_idx = knees_idx  # [left, right]
        self._shlds_frame_idx = shlds_idx  # [left, right]

        # Ordered index tensor for gathering all 11 sites in all_poses layout:
        # [head, pelv, tors, left_foot, right_foot, left_palm, right_palm,
        #  left_knee, right_knee, left_shoulder, right_shoulder]
        self._all_site_idx = torch.tensor(
            [self._head_frame_idx,
             self._pelv_frame_idx,
             self._tors_frame_idx,
             *self._feet_frame_idx,
             *self._hands_frame_idx,
             *self._knees_frame_idx,
             *self._shlds_frame_idx],
            dtype=torch.int32, device=self.device,
        )  # (11,)

        # ------------------------------------------------------------------ robot body indices (for body-frame velocities)
        # Used in _post_physics_step to look up per-body velocities from robot.data.body_*_vel_w.
        pelv_body_idx, _ = self.robot.find_bodies("pelvis")
        tors_body_idx, _ = self.robot.find_bodies("torso_link")
        self._pelv_body_idx = pelv_body_idx[0]   # scalar: index of pelvis in body list
        self._tors_body_idx = tors_body_idx[0]   # scalar: index of torso_link in body list

        # ------------------------------------------------------------------ contact sensors
        # foot_ground_contact tracks ankle_roll_link vs ground (used by rewards/gait and obs)
        # left/right_foot_self_contact detect cross-limb collisions for termination
        self._foot_ground_contact: ContactSensor = self.scene.sensors["foot_ground_contact"]
        self._left_foot_contact:   ContactSensor = self.scene.sensors["left_foot_self_contact"]
        self._right_foot_contact:  ContactSensor = self.scene.sensors["right_foot_self_contact"]

        # ------------------------------------------------------------------ gait constants
        self._init_phase_l  = torch.tensor((0.0, math.pi), dtype=torch.float32, device=self.device)  # (2,)
        self._init_phase_r  = torch.tensor((math.pi, 0.0), dtype=torch.float32, device=self.device)  # (2,)
        self._gait_bound    = cfg.gait_config.gait_bound
        self._stance_phase  = torch.tensor((0.0, 0.0), dtype=torch.float32, device=self.device)      # (2,)
        self._stop_cmd = torch.zeros(4, dtype=torch.float32, device=self.device)

        # ------------------------------------------------------------------ per-env state buffers
        self._init_state_buffers()

    # ------------------------------------------------------------------
    def _init_state_buffers(self):
        """Allocate zero-initialised per-env state tensors."""
        N = self.num_envs
        d = self.device

        self.cmd_state = CommandState(
            command=torch.zeros(N, 4, device=d),
            last_command=torch.zeros(N, 4, device=d),
            command_delay=torch.zeros(N, 4, device=d),
        )
        self.action_state = ActionState(
            motor_targets=self._default_joint_pos.unsqueeze(0).expand(N, -1).clone(),
            last_act=torch.zeros(N, 12, device=d),
            last_last_act=torch.zeros(N, 12, device=d),
        )
        self.vel_state = VelocityState(
            local_lin_vel=torch.zeros(N, 3, device=d),
            global_lin_vel=torch.zeros(N, 3, device=d),
            global_ang_vel=torch.zeros(N, 3, device=d),
            last_joint_vel=torch.zeros(N, 29, device=d),
            last_feet_vel=torch.zeros(N, 2, device=d),
        )
        self.navi_state = NaviFrameState(
            navi2world_rot=torch.eye(3, device=d).unsqueeze(0).expand(N, -1, -1).clone(),
            navi2world_pose=torch.eye(4, device=d).unsqueeze(0).expand(N, -1, -1).clone(),
            navi_torso_rpy=torch.zeros(N, 3, device=d),
            navi_torso_lin_vel=torch.zeros(N, 3, device=d),
            navi_torso_ang_vel=torch.zeros(N, 3, device=d),
            navi_pelvis_rpy=torch.zeros(N, 3, device=d),
            navi_pelvis_lin_vel=torch.zeros(N, 3, device=d),
            navi_pelvis_ang_vel=torch.zeros(N, 3, device=d),
        )
        self.gait_state = GaitState(
            stop_timestep=torch.full((N,), 100, dtype=torch.int32, device=d),
            phase=self._init_phase_l.unsqueeze(0).expand(N, -1).clone(),
            phase_dt=torch.zeros(N, device=d),
            gait_mask=torch.zeros(N, 2, device=d),
            gait_freq=torch.full((N,), self.cfg.gait_config.freq_range[1], device=d),
            foot_height=torch.full((N,), self.cfg.gait_config.foot_height_range[1], device=d),
        )
        self.rand_state = DomainRandState(
            kp_scale=torch.ones(N, device=d),
            kd_scale=torch.ones(N, device=d),
            rfi_lim_scale=torch.zeros(N, 29, device=d),
        )
        self.push_state = PushState(
            push=torch.zeros(N, 2, device=d),
            push_step=torch.zeros(N, dtype=torch.int32, device=d),
            push_interval_steps=torch.full((N,), 250, dtype=torch.int32, device=d),
        )
        self.field_state = FieldState(
            gf=torch.zeros(N, NUM_FIELD_SITES, 3, device=d),
            bf=torch.zeros(N, NUM_FIELD_SITES, 3, device=d),
            df=torch.zeros(N, NUM_FIELD_SITES, 1, device=d),
            gf_delay=torch.zeros(N, NUM_FIELD_SITES, 3, device=d),
            bf_delay=torch.zeros(N, NUM_FIELD_SITES, 3, device=d),
            df_delay=torch.zeros(N, NUM_FIELD_SITES, 1, device=d),
        )
        self.body_state = BodyState(
            head_pos=torch.zeros(N, 3, device=d),
            head_vel=torch.zeros(N, 3, device=d),
            pelv_pos=torch.zeros(N, 3, device=d),
            tors_pos=torch.zeros(N, 3, device=d),
            feet_pos=torch.zeros(N, 2, 3, device=d),
            feet_vel=torch.zeros(N, 2, 3, device=d),
            hands_pos=torch.zeros(N, 2, 3, device=d),
            hands_vel=torch.zeros(N, 2, 3, device=d),
            knees_pos=torch.zeros(N, 2, 3, device=d),
            shlds_pos=torch.zeros(N, 2, 3, device=d),
            odom_delay=torch.zeros(N, 7, device=d),
        )

    # ------------------------------------------------------------------
    def _setup_scene(self):
        super()._setup_scene()
        self.robot = Articulation(self.cfg.scene.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )
        self.scene.clone_environments()
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset selected environments, faithfully following env_cat.py reset().

        env_cat.py equivalent sections:
            qpos / qvel randomisation  → root state + joint state writes
            site_xpos sampling         → FrameTransformer after scene.update()
            field sampling             → self.field_sampler.sample_*
            compute_cmd_from_rtf       → self.compute_cmd_from_rtf()
            gait / domain-rand / push  → per-env state buffer writes
        """
        # 1. Reset scene (restores articulation defaults, clears sensors)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        d = self.device

        # ------------------------------------------------------------------ root state
        # default_root_state: (num_envs, 13)  [pos(3) quat(4) lin_vel(3) ang_vel(3)]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        # Random xy displacement in [-1, 1] m  (matches env_cat.py dxy)
        dxy = torch.empty(n, 2, device=d).uniform_(-1.0, 1.0)
        root_state[:, :2] += dxy

        # Fixed reset height 0.8 m above env floor  (env_cat.py qpos[2] = 0.8)
        root_state[:, 2] = self.scene.env_origins[env_ids, 2] + 0.8

        # Random yaw in [-π/2, π/2]  (env_cat.py: yaw = U(-π/2, π/2))
        yaw = torch.empty(n, device=d).uniform_(-math.pi / 2.0, math.pi / 2.0)
        z_axis = torch.zeros(n, 3, device=d)
        z_axis[:, 2] = 1.0
        delta_q = quat_from_angle_axis(yaw, z_axis)          # (n, 4)  wxyz
        root_state[:, 3:7] = quat_mul(delta_q, root_state[:, 3:7])

        # Random base velocity d(xyzrpy) in [-0.5, 0.5]  (env_cat.py qvel[0:6])
        root_state[:, 7:13] = torch.empty(n, 6, device=d).uniform_(-0.5, 0.5)

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # ------------------------------------------------------------------ joint state
        # env_cat.py: qpos[7:] *= U(0.5, 1.5), clamp to soft limits
        default_jpos = self._default_joint_pos.unsqueeze(0).expand(n, -1).clone()  # (n, 29)
        rand_scale = torch.empty(n, 29, device=d).uniform_(0.5, 1.5)
        joint_pos = (default_jpos * rand_scale).clamp(
            self._soft_lowers.unsqueeze(0),
            self._soft_uppers.unsqueeze(0),
        )
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # ------------------------------------------------------------------ refresh sensors
        # Propagate written state to FrameTransformer so site positions are current
        self.scene.update(dt=0.0)

        # ------------------------------------------------------------------ body positions  (env_cat.py data.site_xpos[...])
        # target_pos_w: (num_envs, num_target_frames, 3)
        all_pos_w = self.frame_transformer.data.target_pos_w[
            env_ids
        ][:, self._all_site_idx, :]                  # (n, 11, 3)

        head_pos, pelv_pos, tors_pos, feet_pos, hands_pos, knees_pos, shlds_pos = torch.split(
            all_pos_w, [1, 1, 1, 2, 2, 2, 2], dim=1
        )
        head_pos = head_pos.squeeze(1)   # (n, 1, 3) -> (n, 3)
        pelv_pos = pelv_pos.squeeze(1)   # (n, 1, 3) -> (n, 3)
        tors_pos = tors_pos.squeeze(1)   # (n, 1, 3) -> (n, 3)

        # ------------------------------------------------------------------ field sampling  (env_cat.py sample_field)
        gf = self.field_sampler.sample_gf(all_pos_w)  # (n, 11, 3)
        bf = self.field_sampler.sample_bf(all_pos_w)  # (n, 11, 3)
        df = self.field_sampler.sample_sdf(all_pos_w) # (n, 11, 1)

        # ------------------------------------------------------------------ command  (env_cat.py compute_cmd_from_rtf)
        pelvgf   = gf[:, 1]                          # (n, 3)
        cgf_init = gf[:, [0, 3, 4, 5, 6]]           # (n, 5, 3)
        cbf_init = bf[:, [0, 3, 4, 5, 6]]           # (n, 5, 3)
        command  = self.compute_cmd_from_rtf(pelvgf, cgf_init, cbf_init)  # (n, 4)

        # ------------------------------------------------------------------ push interval  (env_cat.py push_config)
        push_interval_s = torch.empty(n, device=d).uniform_(
            self.cfg.push_config.interval_range[0],
            self.cfg.push_config.interval_range[1],
        )
        push_interval_steps = (push_interval_s / self._ctrl_dt).round().to(torch.int32)

        # ------------------------------------------------------------------ gait  (env_cat.py gait_config)
        gait_freq = torch.empty(n, device=d).uniform_(
            self.cfg.gait_config.freq_range[0],
            self.cfg.gait_config.freq_range[1],
        )
        phase_dt = 2.0 * math.pi * self._ctrl_dt * gait_freq  # (n,)

        cond     = torch.rand(n, device=d) > 0.5              # (n,) bool
        phase_l  = self._init_phase_l.unsqueeze(0).expand(n, -1)
        phase_r  = self._init_phase_r.unsqueeze(0).expand(n, -1)
        phase    = torch.where(cond.unsqueeze(-1), phase_l, phase_r)  # (n, 2)

        foot_height = torch.empty(n, device=d).uniform_(
            self.cfg.gait_config.foot_height_range[0],
            self.cfg.gait_config.foot_height_range[1],
        )

        # ------------------------------------------------------------------ domain randomisation  (env_cat.py dm_rand_config)
        kp_scale = torch.empty(n, device=d).uniform_(*self.cfg.dm_rand_config.kp_range)
        kd_scale = torch.empty(n, device=d).uniform_(*self.cfg.dm_rand_config.kd_range)

        rfi_noise = torch.empty(n, 29, device=d).uniform_(*self.cfg.dm_rand_config.rfi_lim_range)
        rfi_lim_scale = (
            self.cfg.dm_rand_config.rfi_lim
            * rfi_noise
            * self._torque_limit.unsqueeze(0)
        )  # (n, 29)

        if not self.cfg.dm_rand_config.enable_pd:
            kp_scale.fill_(1.0)
            kd_scale.fill_(1.0)
        if not self.cfg.dm_rand_config.enable_rfi:
            rfi_lim_scale.zero_()

        # ------------------------------------------------------------------ write state buffers
        # FieldState
        self.field_state.gf[env_ids]       = gf
        self.field_state.bf[env_ids]       = bf
        self.field_state.df[env_ids]       = df
        self.field_state.gf_delay[env_ids] = gf.clone()
        self.field_state.bf_delay[env_ids] = bf.clone()
        self.field_state.df_delay[env_ids] = df.clone()

        # BodyState
        self.body_state.head_pos[env_ids]  = head_pos
        self.body_state.head_vel[env_ids]  = torch.zeros_like(head_pos)
        self.body_state.pelv_pos[env_ids]  = pelv_pos
        self.body_state.tors_pos[env_ids]  = tors_pos
        self.body_state.feet_pos[env_ids]  = feet_pos
        self.body_state.feet_vel[env_ids]  = torch.zeros_like(feet_pos)
        self.body_state.hands_pos[env_ids] = hands_pos
        self.body_state.hands_vel[env_ids] = torch.zeros_like(hands_pos)
        self.body_state.knees_pos[env_ids] = knees_pos
        self.body_state.shlds_pos[env_ids] = shlds_pos
        # odom_delay: [x, y, z, qw, qx, qy, qz]  (env_cat.py: odom_delay = qpos[:7])
        self.body_state.odom_delay[env_ids] = root_state[:, :7]

        # CommandState
        self.cmd_state.command[env_ids]       = command
        self.cmd_state.last_command[env_ids]  = torch.zeros(n, 4, device=d)
        self.cmd_state.command_delay[env_ids] = command.clone()

        # ActionState
        self.action_state.motor_targets[env_ids]  = self._default_joint_pos.unsqueeze(0).expand(n, -1).clone()
        self.action_state.last_act[env_ids]       = torch.zeros(n, 12, device=d)
        self.action_state.last_last_act[env_ids]  = torch.zeros(n, 12, device=d)

        # VelocityState
        self.vel_state.local_lin_vel[env_ids]  = torch.zeros(n, 3, device=d)
        self.vel_state.global_lin_vel[env_ids] = torch.zeros(n, 3, device=d)
        self.vel_state.global_ang_vel[env_ids] = torch.zeros(n, 3, device=d)
        self.vel_state.last_joint_vel[env_ids] = torch.zeros(n, 29, device=d)
        self.vel_state.last_feet_vel[env_ids]  = torch.zeros(n, 2, device=d)

        # NaviFrameState
        eye3 = torch.eye(3, device=d).unsqueeze(0).expand(n, -1, -1)
        eye4 = torch.eye(4, device=d).unsqueeze(0).expand(n, -1, -1)
        self.navi_state.navi2world_rot[env_ids]      = eye3
        self.navi_state.navi2world_pose[env_ids]     = eye4
        self.navi_state.navi_torso_rpy[env_ids]      = torch.zeros(n, 3, device=d)
        self.navi_state.navi_torso_lin_vel[env_ids]  = torch.zeros(n, 3, device=d)
        self.navi_state.navi_torso_ang_vel[env_ids]  = torch.zeros(n, 3, device=d)
        self.navi_state.navi_pelvis_rpy[env_ids]     = torch.zeros(n, 3, device=d)
        self.navi_state.navi_pelvis_lin_vel[env_ids] = torch.zeros(n, 3, device=d)
        self.navi_state.navi_pelvis_ang_vel[env_ids] = torch.zeros(n, 3, device=d)

        # GaitState
        self.gait_state.stop_timestep[env_ids] = 100
        self.gait_state.phase[env_ids]         = phase
        self.gait_state.phase_dt[env_ids]      = phase_dt
        self.gait_state.gait_mask[env_ids]     = torch.zeros(n, 2, device=d)
        self.gait_state.gait_freq[env_ids]     = gait_freq
        self.gait_state.foot_height[env_ids]   = foot_height

        # DomainRandState
        self.rand_state.kp_scale[env_ids]      = kp_scale
        self.rand_state.kd_scale[env_ids]      = kd_scale
        self.rand_state.rfi_lim_scale[env_ids] = rfi_lim_scale

        # PushState
        self.push_state.push[env_ids]                = torch.zeros(n, 2, device=d)
        self.push_state.push_step[env_ids]           = 0
        self.push_state.push_interval_steps[env_ids] = push_interval_steps


    # ------------------------------------------------------------------
    # Abstract method stubs required by DirectRLEnv
    # ------------------------------------------------------------------
 
    
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions

        push_theta = torch.empty(self.num_envs, device=self.device).uniform_(0, 2 * math.pi)
        push_magnitude = torch.empty(self.num_envs, device=self.device).uniform_(
            self.cfg.push_config.magnitude_range[0],
            self.cfg.push_config.magnitude_range[1],
        )

        push_signal = (
            (self.push_state.push_step + 1) % self.push_state.push_interval_steps
        ) == 0

        push = torch.stack([torch.cos(push_theta), torch.sin(push_theta)], dim=-1)
        push = push * push_signal.unsqueeze(-1)
        push = push * push_magnitude.unsqueeze(-1)
        # if not self.cfg.push_config.enable:
        #    push.zero_()

        root_vel = self.robot.data.root_vel_w.clone()
        root_vel[:, :2] += push
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=None)

        self.push_state.push = push
        self.push_state.push_step += 1

        lower_motor_targets = torch.clamp(
            self.action_state.motor_targets[:, self._action_joint_ids]
            + actions * self.cfg.action_scale,
            self._soft_lowers[self._action_joint_ids],
            self._soft_uppers[self._action_joint_ids],
        )
        self.action_state.motor_targets[:, self._action_joint_ids] = lower_motor_targets

        self.action_state.last_last_act = self.action_state.last_act.clone()
        self.action_state.last_act = actions.clone()

    def _apply_action(self):
        pos_err = self.action_state.motor_targets - self.robot.data.joint_pos
        vel_err = -self.robot.data.joint_vel
        
        # kp_scale: (num_envs,) -> (num_envs, 1) for broadcasting
        kp = self.rand_state.kp_scale.unsqueeze(-1) * self._kps
        kd = self.rand_state.kd_scale.unsqueeze(-1) * self._kds
        torque = kp * pos_err + kd * vel_err

        # if self.cfg.dm_rand_config.enable_rfi:
        rfi_noise = torch.rand_like(torque) * 2.0 - 1.0
        torque = torque + self.rand_state.rfi_lim_scale * rfi_noise

        torque = torch.clamp(torque, -self._torque_limit, self._torque_limit)
        self.robot.set_joint_effort_target(torque)

    def _get_observations(self) -> dict:
        """Assemble policy (state) and privileged (critic) observations.

        Faithfully mirrors ``env_cat.py _get_obs(data, info, feet_contact)``.

        Policy state  (N, 162):
            noisy_gyro(3) + noisy_gvec(3) + noisy_Δjoint_pos(23) + noisy_joint_vel(23) +
            last_act(12) + motor_targets(12) + command_navi(4) + foot_height(1) +
            gait_phase(4) + pf_navi_delayed(77)

        Privileged state (N, 250):
            IMU+joints(52) + controls+gait(33) + linvel(3) + PF(77) + body(48) + status(6) + domain_rand(31)
        """
        N = self.num_envs
        noise_level  = self.cfg.noise_config.level
        noise_scales = self.cfg.noise_config.scales

        # ------------------------------------------------------------------ IMU (pelvis root)
        gyro_pelvis   = self.robot.data.root_ang_vel_b           # (N, 3) angular vel in body frame
        gvec_pelvis   = self.robot.data.projected_gravity_b      # (N, 3) gravity direction in body frame
        linvel_pelvis = self.robot.data.root_lin_vel_b           # (N, 3) hint: local linear velocity

        # ------------------------------------------------------------------ joint state
        joint_pos_obs = self.robot.data.joint_pos[:, self._obs_joint_ids]   # (N, 23)
        joint_vel_obs = self.robot.data.joint_vel[:, self._obs_joint_ids]   # (N, 23)
        default_obs   = self._default_joint_pos[self._obs_joint_ids]        # (23,)

        # ------------------------------------------------------------------ gait phase
        gait_phase = torch.cat([
            torch.cos(self.gait_state.phase),   # (N, 2)
            torch.sin(self.gait_state.phase),   # (N, 2)
        ], dim=-1)                               # (N, 4)

        # ------------------------------------------------------------------ commands
        command       = self.cmd_state.command.clone()   # (N, 4)
        command_delay = self.cmd_state.command_delay     # (N, 4)

        # ------------------------------------------------------------------ foot-ground contact
        net_forces_ground = self._foot_ground_contact.data.net_forces_w          # (N, 2, 3)
        feet_contact = (torch.norm(net_forces_ground, dim=-1) > self._foot_ground_contact.cfg.force_threshold).float()  # (N, 2)

        # ------------------------------------------------------------------ navi frame
        navi2world_rot = self.navi_state.navi2world_rot          # (N, 3, 3)
        nT = navi2world_rot.transpose(-1, -2)                    # (N, 3, 3)  world→navi rotation

        # ------------------------------------------------------------------ observation noise
        noise_gyro = (2 * torch.rand_like(gyro_pelvis) - 1) * noise_level * noise_scales.gyro
        noise_gvec = (2 * torch.rand_like(gvec_pelvis) - 1) * noise_level * noise_scales.gravity
        noise_jpos = (2 * torch.rand_like(joint_pos_obs) - 1) * noise_level * noise_scales.joint_pos
        noise_jvel = (2 * torch.rand_like(joint_vel_obs) - 1) * noise_level * noise_scales.joint_vel

        noisy_gyro     = gyro_pelvis   + noise_gyro    # (N, 3)
        noisy_gvec     = gvec_pelvis   + noise_gvec    # (N, 3)
        noisy_jpos_obs = joint_pos_obs + noise_jpos    # (N, 23)
        noisy_jvel_obs = joint_vel_obs + noise_jvel    # (N, 23)

        # ------------------------------------------------------------------ navi-frame delayed potential fields
        gf_delay = self.field_state.gf_delay   # (N, 11, 3)
        bf_delay = self.field_state.bf_delay   # (N, 11, 3)
        df_delay = self.field_state.df_delay   # (N, 11, 1)

        gf_delay_navi = torch.bmm(nT, gf_delay.transpose(-1, -2)).transpose(-1, -2)   # (N, 11, 3)
        bf_delay_navi = torch.bmm(nT, bf_delay.transpose(-1, -2)).transpose(-1, -2)   # (N, 11, 3)

        bf_mask           = (df_delay < 0.5).float()           # (N, 11, 1)
        bf_delay_navi_msk = bf_delay_navi * bf_mask            # (N, 11, 3)
        df_delay_clipped  = torch.clamp(df_delay, -1.0, 0.5)  # (N, 11, 1)

        # ------------------------------------------------------------------ navi-frame command for policy obs
        cmd_delay_vel      = command_delay[:, 1:4].unsqueeze(-1)                         # (N, 3, 1)
        cmd_vel_navi       = torch.bmm(nT, cmd_delay_vel).squeeze(-1)                    # (N, 3)
        command_for_policy = command.clone()
        command_for_policy[:, 1:4] = cmd_vel_navi
        command_for_policy[:, 3]   = 0.0 
        # ------------------------------------------------------------------ pf vector for policy obs (N, 77)
        # Ordering mirrors env_cat.py pf hstack:
        #   head(7) + pelv(7) + tors(7) + feet(14) + hands(14) + knees(14) + shlds(14)
        pf = torch.cat([
            # head  (site 0): gf(3) bf(3) df(1)
            gf_delay_navi[:, 0, :].reshape(N, -1),        # (N, 3)
            bf_delay_navi_msk[:, 0, :].reshape(N, -1),    # (N, 3)
            df_delay_clipped[:, 0, :].reshape(N, -1),     # (N, 1)
            # pelv  (site 1)
            gf_delay_navi[:, 1, :].reshape(N, -1),
            bf_delay_navi_msk[:, 1, :].reshape(N, -1),
            df_delay_clipped[:, 1, :].reshape(N, -1),
            # tors  (site 2)
            gf_delay_navi[:, 2, :].reshape(N, -1),
            bf_delay_navi_msk[:, 2, :].reshape(N, -1),
            df_delay_clipped[:, 2, :].reshape(N, -1),
            # feet  (sites 3-4): gf(6) bf(6) df(2)
            gf_delay_navi[:, 3:5, :].reshape(N, -1),
            bf_delay_navi_msk[:, 3:5, :].reshape(N, -1),
            df_delay_clipped[:, 3:5, :].reshape(N, -1),
            # hands (sites 5-6)
            gf_delay_navi[:, 5:7, :].reshape(N, -1),
            bf_delay_navi_msk[:, 5:7, :].reshape(N, -1),
            df_delay_clipped[:, 5:7, :].reshape(N, -1),
            # knees (sites 7-8)
            gf_delay_navi[:, 7:9, :].reshape(N, -1),
            bf_delay_navi_msk[:, 7:9, :].reshape(N, -1),
            df_delay_clipped[:, 7:9, :].reshape(N, -1),
            # shlds (sites 9-10)
            gf_delay_navi[:, 9:11, :].reshape(N, -1),
            bf_delay_navi_msk[:, 9:11, :].reshape(N, -1),
            df_delay_clipped[:, 9:11, :].reshape(N, -1),
        ], dim=-1)  # (N, 77)

        # ------------------------------------------------------------------ policy observation (state)  (N, 162)
        state = torch.cat([
            noisy_gyro,                                                   # 3
            noisy_gvec,                                                   # 3
            noisy_jpos_obs - default_obs.unsqueeze(0),                    # 23
            noisy_jvel_obs,                                               # 23
            self.action_state.last_act,                                   # 12
            self.action_state.motor_targets[:, self._action_joint_ids],   # 12
            command_for_policy,                                           # 4
            self.gait_state.foot_height.unsqueeze(-1),                    # 1
            gait_phase,                                                   # 4
            pf,                                                           # 77
        ], dim=-1)  # (N, 162) 
        # ------------------------------------------------------------------ privileged observation (critic)  (N, 250)
        gf_cur = self.field_state.gf   # (N, 11, 3)
        bf_cur = self.field_state.bf   # (N, 11, 3)
        df_cur = self.field_state.df   # (N, 11, 1)

        privileged_state = torch.cat([
            gyro_pelvis,                                                  # 3
            gvec_pelvis,                                                  # 3
            joint_pos_obs - default_obs.unsqueeze(0),                     # 23
            joint_vel_obs,                                                # 23
            self.action_state.last_act,                                   # 12
            self.action_state.motor_targets[:, self._action_joint_ids],   # 12
            command,                                                      # 4  (world-frame, not navi)
            self.gait_state.foot_height.unsqueeze(-1),                    # 1
            gait_phase,                                                   # 4
            linvel_pelvis,                                                # 3
            # current (non-delayed, non-navi) normalised fields  (77 total)
            gf_cur[:, 0, :].reshape(N, -1),    # headgf  3
            bf_cur[:, 0, :].reshape(N, -1),    # headbf  3
            df_cur[:, 0, :].reshape(N, -1),    # headdf  1
            gf_cur[:, 1, :].reshape(N, -1),    # pelvgf  3
            bf_cur[:, 1, :].reshape(N, -1),    # pelvbf  3
            df_cur[:, 1, :].reshape(N, -1),    # pelvdf  1
            gf_cur[:, 2, :].reshape(N, -1),    # torsgf  3
            bf_cur[:, 2, :].reshape(N, -1),    # torsbf  3
            df_cur[:, 2, :].reshape(N, -1),    # torsdf  1
            gf_cur[:, 3:5, :].reshape(N, -1),  # feetgf  6
            bf_cur[:, 3:5, :].reshape(N, -1),  # feetbf  6
            df_cur[:, 3:5, :].reshape(N, -1),  # feetdf  2
            gf_cur[:, 5:7, :].reshape(N, -1),  # handsgf 6
            bf_cur[:, 5:7, :].reshape(N, -1),  # handsbf 6
            df_cur[:, 5:7, :].reshape(N, -1),  # handsdf 2
            gf_cur[:, 7:9, :].reshape(N, -1),  # kneesgf 6
            bf_cur[:, 7:9, :].reshape(N, -1),  # kneesbf 6
            df_cur[:, 7:9, :].reshape(N, -1),  # kneesdf 2
            gf_cur[:, 9:11, :].reshape(N, -1), # shldsgf 6
            bf_cur[:, 9:11, :].reshape(N, -1), # shldsbf 6
            df_cur[:, 9:11, :].reshape(N, -1), # shldsdf 2
            # body positions and velocities in world frame  (48 total)
            self.body_state.head_pos.reshape(N, -1),      # 3
            self.body_state.head_vel.reshape(N, -1),      # 3
            self.body_state.pelv_pos.reshape(N, -1),      # 3
            self.body_state.tors_pos.reshape(N, -1),      # 3
            self.body_state.feet_pos.reshape(N, -1),      # 6
            self.body_state.feet_vel.reshape(N, -1),      # 6
            self.body_state.hands_pos.reshape(N, -1),     # 6
            self.body_state.hands_vel.reshape(N, -1),     # 6
            self.body_state.knees_pos.reshape(N, -1),     # 6
            self.body_state.shlds_pos.reshape(N, -1),     # 6
            # status  (6 total)
            self.navi_state.navi_torso_rpy[:, :2],        # 2  (roll, pitch)
            self.gait_state.gait_mask,                    # 2
            feet_contact,                                 # 2
            # domain randomisation  (31 total)
            self.rand_state.kp_scale.unsqueeze(-1),       # 1
            self.rand_state.kd_scale.unsqueeze(-1),       # 1
            self.rand_state.rfi_lim_scale,                # 29
        ], dim=-1)  # (N, 250)
        state            = torch.nan_to_num(state)
        privileged_state = torch.nan_to_num(privileged_state)

        return {"policy": state, "critic": privileged_state}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._post_physics_step()

        fall_term: torch.Tensor = self.robot.data.projected_gravity_b[:, 2] > 0.0  # (N,)
        fall_term = fall_term | (self.body_state.head_pos[:, 2] < 0.7)

        lf_forces = self._left_foot_contact.data.force_matrix_w   # (N, 1, 2, 3)
        rf_forces = self._right_foot_contact.data.force_matrix_w  # (N, 1, 2, 3)
        lf_contact = torch.norm(lf_forces, dim=-1).max(dim=-1).values.max(dim=-1).values > 0.1
        rf_contact = torch.norm(rf_forces, dim=-1).max(dim=-1).values.max(dim=-1).values > 0.1
        contact_term: torch.Tensor = lf_contact | rf_contact

        min_df = self.field_state.df[:, :, 0].min(dim=1).values  # (N,)
        contact_term = contact_term | (min_df < -self.cfg.term_collision_threshold)
        contact_term = contact_term & (self.episode_length_buf >= 50)
        joint_pos = self.robot.data.joint_pos[:, self._all_joint_ids]  # (N, 29)
        nan_term: torch.Tensor = (
            torch.isnan(self.robot.data.root_pos_w).any(dim=-1)
            | torch.isnan(joint_pos).any(dim=-1)
        )  # (N,)

        terminated = fall_term | contact_term | nan_term
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated
    
    # ------------------------------------------------------------------
    def _post_physics_step(self):
        """Update all per-env state buffers after physics.  Mirrors env_cat.py step().

        Order mirrors env_cat.py exactly:
          1. velocity state
          2. navi frame
          3. last_command bookkeeping
          4. body positions / velocities (finite-difference)
          5. raw potential-field sampling
          6. command recompute from raw fields
          7. odom delay update
          8. delayed body poses + delayed field sampling
          9. gait phase update (inlined from _update_phase)
         10. GF / BF normalisation
         11. command_delay recompute from normalised delayed fields
         12. field state write-back
         13. episode step counter + last-vel bookkeeping
        """
        N   = self.num_envs
        EPS = 1e-6

        # ------------------------------------------------------------------ 1. velocity state
        self.vel_state.local_lin_vel  = self.robot.data.root_lin_vel_b.clone()   # (N, 3)  pelvis local frame
        self.vel_state.global_lin_vel = self.robot.data.root_lin_vel_w.clone()   # (N, 3)  world frame
        self.vel_state.global_ang_vel = self.robot.data.root_ang_vel_w.clone()   # (N, 3)  world frame

        # ------------------------------------------------------------------ 2. navi frame
        pelv_quat_w = self.frame_transformer.data.target_quat_w[:, self._pelv_frame_idx, :]   # (N, 4) wxyz
        pelv_mat_w  = matrix_from_quat(pelv_quat_w)                                           # (N, 3, 3)

        # navi frame
        pelv_yaw_quat = yaw_quat(pelv_quat_w)  # (N, 4) - yaw component only
        navi2world_rot = matrix_from_quat(pelv_yaw_quat)  # (N, 3, 3)

        # 4×4 navi-to-world pose matrix
        pelv_pos_w = self.frame_transformer.data.target_pos_w[:, self._pelv_frame_idx, :]  # (N, 3)
        navi2world_pose = self.navi_state.navi2world_pose
        navi2world_pose[:, :3, :3] = navi2world_rot
        navi2world_pose[:, :2, 3]  = pelv_pos_w[:, :2]
        navi2world_pose[:, 2,  3]  = self.cfg.reward_config.base_height_target
        navi2world_pose[:, 3,  3]  = 1.0
        self.navi_state.navi2world_rot  = navi2world_rot
        self.navi_state.navi2world_pose = navi2world_pose

        nT = navi2world_rot.transpose(-1, -2)   # (N, 3, 3)

        # pelvis in navi frame
        pelvis2navi_rot  = nT @ pelv_mat_w                          # (N, 3, 3)
        pelv2navi_quat   = quat_from_matrix(pelvis2navi_rot)        # (N, 4) wxyz
        roll, pitch, yaw = euler_xyz_from_quat(pelv2navi_quat)
        self.navi_state.navi_pelvis_rpy = torch.stack([roll, pitch, yaw], dim=-1)  # (N, 3)
        self.navi_state.navi_pelvis_lin_vel = torch.bmm(
            nT, self.vel_state.global_lin_vel.unsqueeze(-1)
        ).squeeze(-1)   # (N, 3)
        self.navi_state.navi_pelvis_ang_vel = torch.bmm(
            nT, self.vel_state.global_ang_vel.unsqueeze(-1)
        ).squeeze(-1)   # (N, 3)

        # torso in navi frame
        tors_quat_w      = self.frame_transformer.data.target_quat_w[:, self._tors_frame_idx, :]  # (N, 4)
        tors_mat_w       = matrix_from_quat(tors_quat_w)                   # (N, 3, 3)
        torso2navi_rot   = nT @ tors_mat_w                                 # (N, 3, 3)
        tors2navi_quat   = quat_from_matrix(torso2navi_rot)                # (N, 4)
        roll, pitch, yaw = euler_xyz_from_quat(tors2navi_quat)
        self.navi_state.navi_torso_rpy = torch.stack([roll, pitch, yaw], dim=-1)  # (N, 3)

        tors_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._tors_body_idx, :]  # (N, 3)
        tors_ang_vel_w = self.robot.data.body_ang_vel_w[:, self._tors_body_idx, :]  # (N, 3)
        self.navi_state.navi_torso_lin_vel = torch.bmm(
            nT, tors_lin_vel_w.unsqueeze(-1)
        ).squeeze(-1)   # (N, 3)
        self.navi_state.navi_torso_ang_vel = torch.bmm(
            nT, tors_ang_vel_w.unsqueeze(-1)
        ).squeeze(-1)   # (N, 3)

        # ------------------------------------------------------------------ 3. last_command bookkeeping
        self.cmd_state.last_command = self.cmd_state.command.clone()   # (N, 4)

        # ------------------------------------------------------------------ 4. body positions / velocities
        all_pos_w = self.frame_transformer.data.target_pos_w[:, self._all_site_idx, :]  # (N, 11, 3)

        head_pos_w, pelv_pos_w, tors_pos_w, feet_pos_w, hands_pos_w, knees_pos_w, shlds_pos_w = torch.split(
            all_pos_w, [1, 1, 1, 2, 2, 2, 2], dim=1
        )
        head_pos_w  = head_pos_w.squeeze(1)   # (N, 1, 3) -> (N, 3)
        pelv_pos_w  = pelv_pos_w.squeeze(1)   # (N, 1, 3) -> (N, 3)
        tors_pos_w  = tors_pos_w.squeeze(1)   # (N, 1, 3) -> (N, 3)

        head_vel  = (head_pos_w  - self.body_state.head_pos)  / self._ctrl_dt   # (N, 3)
        feet_vel  = (feet_pos_w  - self.body_state.feet_pos)  / self._ctrl_dt   # (N, 2, 3)
        hands_vel = (hands_pos_w - self.body_state.hands_pos) / self._ctrl_dt   # (N, 2, 3)

        self.body_state.head_pos  = head_pos_w
        self.body_state.head_vel  = head_vel
        self.body_state.pelv_pos  = pelv_pos_w
        self.body_state.tors_pos  = tors_pos_w
        self.body_state.feet_pos  = feet_pos_w
        self.body_state.feet_vel  = feet_vel
        self.body_state.hands_pos = hands_pos_w
        self.body_state.hands_vel = hands_vel
        self.body_state.knees_pos = knees_pos_w
        self.body_state.shlds_pos = shlds_pos_w

        # ------------------------------------------------------------------ 5. raw field sampling
        gf = self.field_sampler.sample_gf(all_pos_w)    # (N, 11, 3)
        bf = self.field_sampler.sample_bf(all_pos_w)    # (N, 11, 3)
        df = self.field_sampler.sample_sdf(all_pos_w)   # (N, 11, 1)

        # ------------------------------------------------------------------ 6. command from raw fields
        pelvgf = gf[:, 1, :]                        # (N, 3)
        cgf    = gf[:, [0, 3, 4, 5, 6], :]          # (N, 5, 3)
        cbf    = bf[:, [0, 3, 4, 5, 6], :]          # (N, 5, 3)
        command = self.compute_cmd_from_rtf(pelvgf, cgf, cbf)  # (N, 4)
        self.cmd_state.command = command

        # ------------------------------------------------------------------ 7. odom delay update
        update_mask = ((self.episode_length_buf % self.cfg.delay_update_interval) == 0)  # (N,) bool
        current_odom = self.robot.data.root_state_w[:, :7]  # (N, 7)  [x,y,z, qw,qx,qy,qz]
        odom_delay = torch.where(
            update_mask.unsqueeze(-1), current_odom, self.body_state.odom_delay
        )  # (N, 7)
        self.body_state.odom_delay = odom_delay

        # ------------------------------------------------------------------ 8. delayed body poses + fields
        p_gt   = self.robot.data.root_pos_w    # (N, 3)
        q_gt   = self.robot.data.root_quat_w   # (N, 4) wxyz
        p_odom = odom_delay[:, :3]             # (N, 3)
        q_odom = odom_delay[:, 3:7]            # (N, 4) wxyz

        relative_pos    = all_pos_w - p_gt.unsqueeze(1)                                   # (N, 11, 3)
        q_gt_exp        = q_gt.unsqueeze(1).expand(-1, NUM_FIELD_SITES, -1)               # (N, 11, 4)
        q_odom_exp      = q_odom.unsqueeze(1).expand(-1, NUM_FIELD_SITES, -1)             # (N, 11, 4)
        body_pos_local  = quat_apply_inverse(q_gt_exp, relative_pos)                      # (N, 11, 3)
        all_poses_delay = p_odom.unsqueeze(1) + quat_apply(q_odom_exp, body_pos_local)    # (N, 11, 3)

        gf_delay = self.field_sampler.sample_gf(all_poses_delay)   # (N, 11, 3)
        bf_delay = self.field_sampler.sample_bf(all_poses_delay)   # (N, 11, 3)
        df_delay = self.field_sampler.sample_sdf(all_poses_delay)  # (N, 11, 1)

        # ------------------------------------------------------------------ 9. gait phase update  (inlined _update_phase)
        task_mask      = self.cmd_state.command[:, 0]       # move_flag after recompute  (N,)
        last_task_mask = self.cmd_state.last_command[:, 0]  # previous move_flag          (N,)

        stop_ts     = self.gait_state.stop_timestep.float()   # (N,)
        before_stop = stop_ts > 50                            # (N,)
        during_stop = (~before_stop) & (stop_ts > 0)          # (N,)
        after_stop  = (~before_stop) & (~during_stop)         # (N,)

        move2stop = (last_task_mask == 1.0) & (task_mask == 0.0) & before_stop
        stop_ts   = torch.where(move2stop, torch.full_like(stop_ts, 50.0), stop_ts)
        stop_ts   = torch.where(during_stop, stop_ts - 1.0, stop_ts)
        self.gait_state.stop_timestep = stop_ts.to(torch.int32)

        command = torch.where(
            before_stop.unsqueeze(-1),
            self.cmd_state.command,
            self._stop_cmd.unsqueeze(0).expand(N, -1),
        )  # (N, 4)
        command = command.clone()
        command[:, 0] = torch.where(after_stop, torch.zeros_like(command[:, 0]), torch.ones_like(command[:, 0]))
        self.cmd_state.command = command

        # advance phase
        phase = self.gait_state.phase + self.gait_state.phase_dt.unsqueeze(-1)   # (N, 2)
        phase = torch.fmod(phase + math.pi, 2.0 * math.pi) - math.pi
        phase = torch.where(
            after_stop.unsqueeze(-1),
            self._stance_phase.unsqueeze(0).expand(N, -1),
            phase,
        )
        self.gait_state.phase = phase

        gait_cycle = torch.cos(phase)                                               # (N, 2)
        gait_mask  = torch.where(gait_cycle >  self._gait_bound,  1.0, 0.0)
        gait_mask  = torch.where(gait_cycle < -self._gait_bound, -1.0, gait_mask)
        self.gait_state.gait_mask = gait_mask

        # ------------------------------------------------------------------ 10. GF / BF normalisation
        move_flag = self.cmd_state.command[:, 0].unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)

        gf_norm = gf * (move_flag > 0.5) / (torch.linalg.norm(gf, dim=-1, keepdim=True) + EPS)
        bf_norm = bf / (torch.linalg.norm(bf, dim=-1, keepdim=True) + EPS)

        gf_delay_norm = gf_delay * (move_flag > 0.5) / (
            torch.linalg.norm(gf_delay, dim=-1, keepdim=True) + EPS
        )
        bf_delay_norm = bf_delay / (torch.linalg.norm(bf_delay, dim=-1, keepdim=True) + EPS)

        # ------------------------------------------------------------------ 11. command_delay
        pelvgf_delay = gf_delay_norm[:, 1, :]                   # (N, 3)
        cgf_delay    = gf_delay_norm[:, [0, 3, 4, 5, 6], :]     # (N, 5, 3)
        cbf_delay    = bf_delay_norm[:, [0, 3, 4, 5, 6], :]     # (N, 5, 3)
        command_delay = self.compute_cmd_from_rtf(pelvgf_delay, cgf_delay, cbf_delay)  # (N, 4)
        self.cmd_state.command_delay = command_delay

        # ------------------------------------------------------------------ 12. field state write-back
        self.field_state.gf       = gf_norm
        self.field_state.bf       = bf_norm
        self.field_state.df       = df
        self.field_state.gf_delay = gf_delay_norm
        self.field_state.bf_delay = bf_delay_norm
        self.field_state.df_delay = df_delay

        self.vel_state.last_joint_vel = self.robot.data.joint_vel[:, self._all_joint_ids].clone()
        self.vel_state.last_feet_vel = feet_vel[:, :, 2]


    def compute_cmd_from_rtf(
        self,
        root_guidance_field: torch.Tensor,
        contact_guidance_field: torch.Tensor,
        contact_boundary_field: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity command from potential-field values.

        Mirrors env_cat.py ``compute_cmd_from_rtf()``. This implements a single
        iteration of field projection: starts with an initial velocity from the
        root guidance field, then adjusts it based on boundary constraints from
        contact points (head, feet, hands).

        Args:
            root_guidance_field: Pelvis guidance-field vector, shape (n, 3).
            contact_guidance_field: Combined guidance field for head+feet+hands,
                shape (n, 5, 3).
            contact_boundary_field: Combined boundary field for head+feet+hands,
                shape (n, 5, 3).

        Returns:
            Command ``[move_flag, vx, vy, vyaw]``, shape (n, 4).
        """
        num_envs = root_guidance_field.shape[0]
        eps = 1e-9

        # Initial velocity from root guidance field (xy component only), scaled
        initial_velocity = root_guidance_field[:, :2] * 0.7  # (n, 2)

        # Normalize boundary field to get obstacle directions (xy plane only)
        boundary_norm_xy = torch.linalg.norm(
            contact_boundary_field[:, :, :2], dim=-1, keepdim=True
        )  # (n, 5, 1)
        boundary_direction = contact_boundary_field[:, :, :2] / (boundary_norm_xy + eps)
        # (n, 5, 2)

        # Project guidance field onto boundary directions
        guidance_projection = torch.sum(
            boundary_direction * contact_guidance_field[:, :, :2], dim=-1
        )  # (n, 5)

        # Project current velocity onto boundary directions
        velocity_projection = torch.sum(
            boundary_direction * initial_velocity.unsqueeze(1), dim=-1
        )  # (n, 5)

        # Compute how much we need to adjust velocity along each boundary direction
        projection_diff = guidance_projection - velocity_projection  # (n, 5)
        boundary_direction_norm_sq = torch.sum(
            boundary_direction * boundary_direction, dim=-1
        )  # (n, 5)
        correction_magnitude = projection_diff / (boundary_direction_norm_sq + eps)
        # (n, 5)

        # Build the correction vectors
        correction_vector = correction_magnitude.unsqueeze(-1) * boundary_direction
        # (n, 5, 2)

        # Only apply correction where guidance projection exceeds velocity projection
        needs_correction = (guidance_projection > velocity_projection).unsqueeze(-1)
        # (n, 5, 1)
        correction_vector = torch.where(
            needs_correction, correction_vector, torch.zeros_like(correction_vector)
        )

        # Apply mean correction across all contact points
        corrected_velocity = initial_velocity + correction_vector.mean(dim=1)
        # (n, 2)

        # Build full command: [move_flag, vx, vy, vyaw], scaled
        command = torch.zeros(num_envs, 4, device=root_guidance_field.device)
        command[:, 0] = 1.0  # move_flag
        command[:, 1] = corrected_velocity[:, 0]  # vx
        command[:, 2] = corrected_velocity[:, 1]  # vy
        command[:, 3] = 0.0  # vyaw
        command = command * 0.75

        # Stop command if velocity magnitude is too small
        velocity_magnitude = torch.linalg.norm(command[:, 1:4], dim=-1)
        stop_mask = velocity_magnitude < 0.2
        command = torch.where(
            stop_mask.unsqueeze(-1),
            self._stop_cmd.unsqueeze(0).expand(num_envs, -1),
            command,
        )

        return command

#NOTE: Use where to replace if