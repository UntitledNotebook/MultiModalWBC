import math as _math

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_control.assets import ASSET_DIR

# ==============================================================================
# Step 1.2 — Body Collision Groups
# Used for SDF-based collision termination (ContactSensor / FrameTransformer).
# Bodies correspond to the MuJoCo sites actually sampled in env_cat.py:
#   headdf   → "head" site on torso_link
#   pelvdf   → "imu_in_pelvis" site on pelvis
#   torsdf   → "imu_in_torso" site on torso_link
#   feetdf   → "left_foot"/"right_foot" sites on ankle_roll_links
#   handsdf  → "left_palm"/"right_palm" sites on wrist_yaw_links
#   kneesdf  → "left_knee"/"right_knee" sites on knee_links
#   shldsdf  → "left_shoulder"/"right_shoulder" sites on shoulder_pitch_links
# ==============================================================================

HEAD_BODIES     = ["torso_link"]
PELVIS_BODIES   = ["pelvis"]
TORSO_BODIES    = ["torso_link"]
FEET_BODIES     = ["left_ankle_roll_link", "right_ankle_roll_link"]
HAND_BODIES     = ["left_wrist_yaw_link",  "right_wrist_yaw_link"]
KNEE_BODIES     = ["left_knee_link",       "right_knee_link"]
SHOULDER_BODIES = ["left_shoulder_pitch_link", "right_shoulder_pitch_link"]

# G1CatEnv  (student)  — all 7 groups, enabled after step ≥ 50
STUDENT_COLLISION_BODIES = (
    HEAD_BODIES + FEET_BODIES + HAND_BODIES
    + KNEE_BODIES + SHOULDER_BODIES + PELVIS_BODIES + TORSO_BODIES
)
# G1CatPriEnv (teacher) — 3 groups only, enabled after step ≥ 100
PRIVILEGED_COLLISION_BODIES = HEAD_BODIES + FEET_BODIES + HAND_BODIES

# ==============================================================================
# FrameTransformer frame names for each body-site group.
# These mirror the MuJoCo site IDs used in env_cat.py.
# ==============================================================================

# Individual group names (match the FrameTransformerCfg.FrameCfg "name" fields)
HEAD_FRAME_NAME   = "head"
PELVIS_FRAME_NAME = "imu_in_pelvis"
TORSO_FRAME_NAME  = "imu_in_torso"
FEET_FRAME_NAMES  = ["left_foot",      "right_foot"]
HANDS_FRAME_NAMES = ["left_palm",      "right_palm"]
KNEES_FRAME_NAMES = ["left_knee",      "right_knee"]
SHLDS_FRAME_NAMES = ["left_shoulder",  "right_shoulder"]

# Ordered list used to build all_poses (matches env_cat.py step() layout):
#   idx 0     : head
#   idx 1     : pelv  (imu_in_pelvis)
#   idx 2     : tors  (imu_in_torso)
#   idx 3-4   : feet  (left_foot, right_foot)
#   idx 5-6   : hands (left_palm, right_palm)
#   idx 7-8   : knees (left_knee, right_knee)
#   idx 9-10  : shlds (left_shoulder, right_shoulder)
ALL_FRAME_NAMES = [
    HEAD_FRAME_NAME,
    PELVIS_FRAME_NAME,
    TORSO_FRAME_NAME,
    *FEET_FRAME_NAMES,
    *HANDS_FRAME_NAMES,
    *KNEES_FRAME_NAMES,
    *SHLDS_FRAME_NAMES,
]
NUM_FIELD_SITES = len(ALL_FRAME_NAMES)  # 11

SITE_BODY_OFFSETS = {
    # Field-sampling / observation sites
    "head":           ("torso_link",                 ( 0.0,      0.0,      0.4    )),
    "left_foot":      ("left_ankle_roll_link",       ( 0.04,     0.0,     -0.037  )),
    "right_foot":     ("right_ankle_roll_link",      ( 0.04,     0.0,     -0.037  )),
    "left_palm":      ("left_wrist_yaw_link",        ( 0.08,     0.0,      0.0    )),
    "right_palm":     ("right_wrist_yaw_link",       ( 0.08,     0.0,      0.0    )),
    # Collision / field-sampling sites
    "left_knee":      ("left_knee_link",             ( 0.0,      0.0,      0.0    )),
    "right_knee":     ("right_knee_link",            ( 0.0,      0.0,      0.0    )),
    "left_shoulder":  ("left_shoulder_pitch_link",   ( 0.0,      0.0,      0.0    )),
    "right_shoulder": ("right_shoulder_pitch_link",  ( 0.0,      0.0,      0.0    )),
    # IMU sites (from XML: <site name="..." pos="..."/>)
    "imu_in_pelvis":  ("pelvis",                     ( 0.04525,  0.0,     -0.08339)),
    "imu_in_torso":   ("torso_link",                 (-0.03959, -0.00224,  0.14792)),
}

# ==============================================================================
# 12 actuated leg joints
ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# 23 observed joints
OBS_JOINT_NAMES = [
    # legs (12)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left arm (4)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    # right arm (4)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]


ALL_JOINT_NAMES = [
    # legs (12)
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left arm (7)
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "left_wrist_roll_joint",      "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_roll_joint",     "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Armature values (from MJCF)
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.01017752004
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# Stiffness (KPs) and Damping (KDs) values from cat_ppo MuJoCo training
# Legs
STIFFNESS_HIP_PITCH = 100.0
STIFFNESS_HIP_ROLL = 100.0
STIFFNESS_HIP_YAW = 100.0
STIFFNESS_KNEE = 200.0
STIFFNESS_ANKLE_PITCH = 80.0
STIFFNESS_ANKLE_ROLL = 20.0

DAMPING_HIP_PITCH = 2.0
DAMPING_HIP_ROLL = 2.0
DAMPING_HIP_YAW = 2.0
DAMPING_KNEE = 4.0
DAMPING_ANKLE_PITCH = 2.0
DAMPING_ANKLE_ROLL = 1.0

# Waist
STIFFNESS_WAIST_YAW = 300.0
STIFFNESS_WAIST_ROLL = 300.0
STIFFNESS_WAIST_PITCH = 300.0

DAMPING_WAIST_YAW = 10.0
DAMPING_WAIST_ROLL = 10.0
DAMPING_WAIST_PITCH = 10.0

# Arms
STIFFNESS_SHOULDER_PITCH = 90.0
STIFFNESS_SHOULDER_ROLL = 60.0
STIFFNESS_SHOULDER_YAW = 20.0
STIFFNESS_ELBOW = 60.0
STIFFNESS_WRIST_ROLL = 20.0
STIFFNESS_WRIST_PITCH = 20.0
STIFFNESS_WRIST_YAW = 20.0

DAMPING_SHOULDER_PITCH = 2.0
DAMPING_SHOULDER_ROLL = 2.0
DAMPING_SHOULDER_YAW = 1.0
DAMPING_ELBOW = 1.0
DAMPING_WRIST_ROLL = 1.0
DAMPING_WRIST_PITCH = 1.0
DAMPING_WRIST_YAW = 1.0

KPS = [
    STIFFNESS_HIP_PITCH,      STIFFNESS_HIP_ROLL,      STIFFNESS_HIP_YAW,
    STIFFNESS_KNEE,           STIFFNESS_ANKLE_PITCH,   STIFFNESS_ANKLE_ROLL,
    STIFFNESS_HIP_PITCH,      STIFFNESS_HIP_ROLL,      STIFFNESS_HIP_YAW,
    STIFFNESS_KNEE,           STIFFNESS_ANKLE_PITCH,   STIFFNESS_ANKLE_ROLL,
    STIFFNESS_WAIST_YAW,      STIFFNESS_WAIST_ROLL,    STIFFNESS_WAIST_PITCH,
    STIFFNESS_SHOULDER_PITCH, STIFFNESS_SHOULDER_ROLL,
    STIFFNESS_SHOULDER_YAW,   STIFFNESS_ELBOW,
    STIFFNESS_WRIST_ROLL,     STIFFNESS_WRIST_PITCH,   STIFFNESS_WRIST_YAW,
    STIFFNESS_SHOULDER_PITCH, STIFFNESS_SHOULDER_ROLL,
    STIFFNESS_SHOULDER_YAW,   STIFFNESS_ELBOW,
    STIFFNESS_WRIST_ROLL,     STIFFNESS_WRIST_PITCH,   STIFFNESS_WRIST_YAW,
]

KDS = [
    DAMPING_HIP_PITCH,      DAMPING_HIP_ROLL,      DAMPING_HIP_YAW,
    DAMPING_KNEE,           DAMPING_ANKLE_PITCH,   DAMPING_ANKLE_ROLL,
    DAMPING_HIP_PITCH,      DAMPING_HIP_ROLL,      DAMPING_HIP_YAW,
    DAMPING_KNEE,           DAMPING_ANKLE_PITCH,   DAMPING_ANKLE_ROLL,
    DAMPING_WAIST_YAW,      DAMPING_WAIST_ROLL,    DAMPING_WAIST_PITCH,
    DAMPING_SHOULDER_PITCH, DAMPING_SHOULDER_ROLL,
    DAMPING_SHOULDER_YAW,   DAMPING_ELBOW,
    DAMPING_WRIST_ROLL,     DAMPING_WRIST_PITCH,   DAMPING_WRIST_YAW,
    DAMPING_SHOULDER_PITCH, DAMPING_SHOULDER_ROLL,
    DAMPING_SHOULDER_YAW,   DAMPING_ELBOW,
    DAMPING_WRIST_ROLL,     DAMPING_WRIST_PITCH,   DAMPING_WRIST_YAW,
]

# Per-joint torque limits — same order as ALL_JOINT_NAMES (29 values)
# Sourced from effort_limit_sim in the actuator configs below.
TORQUE_LIMIT = [
    88.0,  139.0,  88.0,  139.0,  50.0,  50.0,   # left leg
    88.0,  139.0,  88.0,  139.0,  50.0,  50.0,   # right leg
    88.0,   50.0,  50.0,                          # waist (yaw, roll, pitch)
    25.0,   25.0,  25.0,  25.0, 25.0, 5.0, 5.0,  # left arm
    25.0,   25.0,  25.0,  25.0, 25.0, 5.0, 5.0,  # right arm
]
# fmt: on

assert len(ALL_JOINT_NAMES) == 29
assert len(KPS) == 29
assert len(KDS) == 29
assert len(TORQUE_LIMIT) == 29

G1_CAT_CFG = ArticulationCfg(
    
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ), #? Do we have to change this, compare with isaaclab_assets
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            # legs
            ".*_hip_pitch_joint": -0.1,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_ankle_roll_joint": 0.0,
            # waist
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_yaw_joint": 0.0,
            # left arm
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.3,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.28,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            # right arm
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.3,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.28,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness=0.0,  # PD computed manually in env
            damping=0.0,    # PD computed manually in env
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": IdealPDActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=0.0,  # PD computed manually in env
            damping=0.0,    # PD computed manually in env
            armature=2 * ARMATURE_5020,
        ),
        "waist": IdealPDActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=0.0,  # PD computed manually in env
            damping=0.0,    # PD computed manually in env
            armature=2 * ARMATURE_5020,
        ),
        "waist_yaw": IdealPDActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=0.0,  # PD computed manually in env
            damping=0.0,    # PD computed manually in env
            armature=ARMATURE_7520_14,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness=0.0,  # PD computed manually in env
            damping=0.0,    # PD computed manually in env
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)
