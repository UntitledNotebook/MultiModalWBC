from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer import OffsetCfg

from .constants import G1_CAT_CFG, SITE_BODY_OFFSETS


# ==============================================================================
# Nested sub-configs (mirrors config_dict.create(...) in env_cat.py)
# ==============================================================================

@configclass
class GaitCfg:
    """Gait oscillator parameters."""
    gait_bound: float = 0.6
    freq_range: tuple[float, float] = (1.3, 1.5)
    foot_height_range: tuple[float, float] = (0.07, 0.07)


@configclass
class DomainRandCfg:
    """PD-gain and random-force-injection (RFI) domain randomisation."""
    enable_pd: bool = True
    kp_range: tuple[float, float] = (0.75, 1.25)
    kd_range: tuple[float, float] = (0.75, 1.25)
    enable_rfi: bool = True
    rfi_lim: float = 0.1
    rfi_lim_range: tuple[float, float] = (0.5, 1.5)
    enable_ctrl_delay: bool = False
    ctrl_delay_range: tuple[int, int] = (0, 2)


@configclass
class NoiseScalesCfg:
    """Per-channel noise amplitudes (applied as ± uniform)."""
    joint_pos: float = 0.03
    joint_vel: float = 1.5
    gravity: float = 0.05
    gyro: float = 0.2


@configclass
class NoiseCfg:
    """Observation noise configuration."""
    level: float = 1.0   # set to 0.0 to disable
    scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class RewardScalesCfg:
    """Individual reward / cost term weights."""
    # behaviour rewards
    tracking_orientation: float = 2.0
    tracking_root_field: float = 1.0
    body_motion: float = -0.5
    body_rotation: float = 1.0
    foot_contact: float = -1.0
    foot_clearance: float = -15.0
    foot_slip: float = -0.5
    foot_balance: float = -30.0
    foot_far: float = 0.0
    straight_knee: float = -30.0
    # energy / regularisation
    smoothness_joint: float = -1e-6
    smoothness_action: float = -1e-3
    joint_limits: float = -1.0
    joint_torque: float = -1e-4
    # potential field terms
    headgf: float = 0.0
    handsgf: float = 0.0
    feetgf: float = 0.0
    headdf: float = 0.0
    handsdf: float = 0.0
    feetdf: float = 0.0
    kneesdf: float = 0.0
    shldsdf: float = 0.0


@configclass
class RewardCfg:
    """Reward configuration."""
    scales: RewardScalesCfg = RewardScalesCfg()
    base_height_target: float = 0.75
    foot_height_stance: float = 0.0


@configclass
class PushCfg:
    """External push perturbation settings."""
    enable: bool = True
    interval_range: tuple[float, float] = (5.0, 10.0)
    magnitude_range: tuple[float, float] = (0.1, 1.0)


@configclass
class CommandCfg:
    """Velocity command sampling settings."""
    resampling_time: float = 10.0
    stop_prob: float = 0.2


@configclass
class PfCfg:
    """Potential-field (navigation field) configuration."""
    path: str = "data/assets/TypiObs/empty"
    dx: float = 0.04
    # World-frame origin of the loaded field grid (x, y, z)
    origin: tuple[float, float, float] = (-0.5, -1.0, 0.0)


# ==============================================================================
# Scene configuration
# ==============================================================================

@configclass
class G1CatSceneCfg(InteractiveSceneCfg):
    """Scene with G1 robot, ground plane, and sensors."""

    num_envs: int = 4096
    env_spacing: float = 2.5

    # Robot
    robot: ArticulationCfg = G1_CAT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # FrameTransformer: track body sites using offsets from SITE_BODY_OFFSETS
    frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="head",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['head'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['head'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="left_foot",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['left_foot'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['left_foot'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="right_foot",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['right_foot'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['right_foot'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="left_palm",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['left_palm'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['left_palm'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="right_palm",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['right_palm'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['right_palm'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="imu_in_pelvis",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['imu_in_pelvis'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['imu_in_pelvis'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="imu_in_torso",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['imu_in_torso'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['imu_in_torso'][1]),
            ),
            # Zero-offset sites (for collision check / convenience)
            FrameTransformerCfg.FrameCfg(
                name="left_knee",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['left_knee'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['left_knee'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="right_knee",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['right_knee'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['right_knee'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="left_shoulder",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['left_shoulder'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['left_shoulder'][1]),
            ),
            FrameTransformerCfg.FrameCfg(
                name="right_shoulder",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS['right_shoulder'][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS['right_shoulder'][1]),
            ),
        ],
    )

    foot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*ankle_roll_link",
        history_length=1,
        update_period=0.0,
        track_air_time=True,
        force_threshold=1.0,
        filter_prim_paths_expr=["/World/ground"],
    )

    left_foot_self_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        history_length=1,
        update_period=0.0,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/right_ankle_roll_link",       # left_foot ↔ right_foot
            "{ENV_REGEX_NS}/Robot/right_shoulder_pitch_link",   # left_foot ↔ right_shoulder
        ],
        force_threshold=1.0,
    )

    right_foot_self_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        history_length=1,
        update_period=0.0,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/left_ankle_roll_link",        # right_foot ↔ left_foot
            "{ENV_REGEX_NS}/Robot/left_shoulder_pitch_link",    # right_foot ↔ left_shoulder
        ],
        force_threshold=1.0,
    )


# ==============================================================================
# Top-level environment configuration
# ==============================================================================

@configclass
class G1CatEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 Cat student environment (Direct workflow).

    Timing:
        sim_dt  = 0.002 s  (500 Hz physics)
        ctrl_dt = 0.020 s  (50 Hz policy)  → decimation = 10
        episode = 1 000 policy steps        → episode_length_s = 20 s
    """

    # ------------------------------------------------------------------
    # DirectRLEnvCfg MISSING fields (must be provided)
    # ------------------------------------------------------------------

    # Physics simulation: dt=0.002 s matches original sim_dt
    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=10)

    # Scene
    scene: G1CatSceneCfg = G1CatSceneCfg(num_envs=4096, env_spacing=2.5)

    # Timing
    decimation: int = 10                  # ctrl_dt / sim_dt = 0.02 / 0.002
    episode_length_s: float = 20.0        # 1000 steps × 0.02 s

    # Observation / action spaces
    # student obs = 162, privileged (critic) state = 250, actions = 12
    # privileged breakdown: 52+33+3+77+48+6+31 = 250
    #   IMU+joints(52) + controls+gait(33) + linvel(3) + PF(77) + body(48) + status(6) + domain_rand(31)
    observation_space: int = 162
    state_space: int = 250
    action_space: int = 12

    # ------------------------------------------------------------------
    # Environment hyper-parameters (from env_cat.py env_config)
    # ------------------------------------------------------------------

    action_scale: float = 0.5
    history_len: int = 15
    restricted_joint_range: bool = False
    soft_joint_pos_limit_factor: float = 0.95

    # Velocity command ranges [min, max]
    lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_yaw: tuple[float, float] = (-0.5, 0.5)
    # Torso height range [min, max]; upper bound = DEFAULT_CHEST_Z = 1.0
    torso_height: tuple[float, float] = (0.5, 1.0)

    # SDF-based collision termination threshold (metres)
    term_collision_threshold: float = 0.04

    # Delayed odometry update interval (steps)
    delay_update_interval: int = 5

    # ------------------------------------------------------------------
    # Sub-configs
    # ------------------------------------------------------------------

    gait_config: GaitCfg = GaitCfg()
    dm_rand_config: DomainRandCfg = DomainRandCfg()
    noise_config: NoiseCfg = NoiseCfg()
    reward_config: RewardCfg = RewardCfg()
    push_config: PushCfg = PushCfg()
    command_config: CommandCfg = CommandCfg()
    pf_config: PfCfg = PfCfg()
