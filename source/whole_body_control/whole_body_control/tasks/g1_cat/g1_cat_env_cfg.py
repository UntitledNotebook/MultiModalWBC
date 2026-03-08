from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .constants import G1_CAT_CFG, SITE_BODY_OFFSETS, ASSET_DIR, ALL_FRAME_NAMES

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
    path: str = f"{ASSET_DIR}/data/assets/TypiObs/empty/"
    dx: float = 0.04
    # World-frame origin of the loaded field grid (x, y, z)
    origin: tuple[float, float, float] = (-0.5, -1.0, 0.0)


@configclass
class G1CatSceneCfg(InteractiveSceneCfg):
    """Scene configuration for G1 Cat environment."""
    # ground terrain 
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # robot
    robot: ArticulationCfg = G1_CAT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    # frame transformer
    frame_transformer: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name=n,
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SITE_BODY_OFFSETS[n][0]}",
                offset=OffsetCfg(pos=SITE_BODY_OFFSETS[n][1]),
            )
            for n in ALL_FRAME_NAMES
        ] + [
            FrameTransformerCfg.FrameCfg(
                name="left_ankle_roll",
                prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                name="right_ankle_roll",
                prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )
    # contact sensors
    foot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*ankle_roll_link",
        history_length=1,
        update_period=0.0,
        track_air_time=True,
        force_threshold=1.0,
    ) # NOTE: No filter_prim_paths_expr is set here, so this sensor measures the total net contact force on the ankle links from ALL contact surfaces (ground + obstacles). This is NOT the same as pure foot-ground contact — obstacle contact forces are included.
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
    # Obstacle
    obstacle: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/data/assets/TypiObs/empty/obs.usd",
            # Recommended: make it truly static (no dynamics, no gravity)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
                max_depenetration_velocity=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, -1.0, 0.0)),
    ) # Issue: the usd conversion should use proper parameters to disable dynamics


@configclass
class G1CatEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 Cat student environment (Direct workflow).
    """

    # ------------------------------------------------------------------
    # DirectRLEnvCfg MISSING fields (must be provided)
    # ------------------------------------------------------------------

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=10)
    # Scene 
    scene: G1CatSceneCfg = G1CatSceneCfg(num_envs=64, env_spacing=6.0, replicate_physics=True)

    # Timing
    decimation: int = 10                  # ctrl_dt / sim_dt = 0.02 / 0.002
    episode_length_s: float = 20.0        # 1000 steps × 0.02 s

    # Observation / action spaces
    observation_space: int = 162
    state_space: int = 250
    action_space: int = 12

    # ------------------------------------------------------------------
    # Environment hyper-parameters (from env_cat.py env_config)
    # ------------------------------------------------------------------

    action_scale: float = 0.5
    history_len: int = 15
    restricted_joint_range: bool = False
    lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    ang_vel_yaw: tuple[float, float] = (-0.5, 0.5)
    torso_height: tuple[float, float] = (0.5, 1.0)
    term_collision_threshold: float = 0.04
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
