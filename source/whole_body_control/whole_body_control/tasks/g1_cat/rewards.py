from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def reward_tracking_lin_vel(cmd_vel: torch.Tensor, local_lin_vel: torch.Tensor) -> torch.Tensor:
    """Reward for tracking the linear velocity command.
    """
    lin_vel_error = torch.sum(torch.square(cmd_vel[:, :2] - local_lin_vel[:, :2]), dim=-1)
    return torch.exp(-4.0 * lin_vel_error)


def reward_tracking_ang_vel(cmd_vel: torch.Tensor, local_ang_vel: torch.Tensor) -> torch.Tensor:
    """Reward for tracking the angular velocity (yaw) command.
    """
    angvel_error = torch.square(cmd_vel[:, 2] - local_ang_vel[:, 2])
    return torch.exp(-4.0 * angvel_error)


def reward_base_height(root_height: torch.Tensor, target_height: float, move_flag: torch.Tensor) -> torch.Tensor:
    """Reward for maintaining the target base height.
    """
    base_height_error = torch.abs(root_height - target_height)
    rew = torch.exp(-3.0 * base_height_error)
    # Give a fixed reward if idling and high enough
    rew = torch.where((root_height * (1.0 - move_flag)) > target_height, 0.5, rew)
    return rew


def reward_tracking_root_field(cmd_vel: torch.Tensor, local_lin_vel: torch.Tensor) -> torch.Tensor:
    """Reward for tracking the root guidance field velocity command.
    
    Args:
        cmd_vel: Guidance field velocity command [num_envs, 2] (x, y)
        local_lin_vel: Current local linear velocity [num_envs, 2] (x, y)
    """
    lin_vel_error = torch.sum(torch.square(cmd_vel[:, :2] - local_lin_vel[:, :2]), dim=-1)
    return torch.exp(-4.0 * lin_vel_error)

def cost_body_motion(
    local_lin_vel: torch.Tensor, 
    local_ang_vel: torch.Tensor, 
    cmd_vel: torch.Tensor
) -> torch.Tensor:
    """Penalty for unwanted body motions (orthogonal velocity and angular velocity).
    
    Args:
        local_lin_vel: Global linear velocity [num_envs, 3]
        local_ang_vel: Local angular velocity [num_envs, 3]
        cmd_vel: Velocity command [num_envs, 2] (x, y)
    """
    cmd_xy = cmd_vel[:, :2]
    cmd_norm = torch.linalg.norm(cmd_xy, dim=-1, keepdim=True)
    is_zero_cmd = cmd_norm < 1e-6
    cmd_dir = torch.where(is_zero_cmd, torch.zeros_like(cmd_xy), cmd_xy / (cmd_norm + 1e-6))

    lin_xy = local_lin_vel[:, :2]
    # Projected orthogonal velocity
    lin_xy_orth = lin_xy - torch.sum(lin_xy * cmd_dir, dim=-1, keepdim=True) * cmd_dir
    cost_lin_xy_orth = torch.where(is_zero_cmd.squeeze(-1), torch.zeros_like(lin_xy_orth.sum(-1)), torch.sum(torch.square(lin_xy_orth), dim=-1))

    cost = (
        1.2 * cost_lin_xy_orth
        + 0.4 * torch.abs(local_ang_vel[:, 0])
        + 0.4 * torch.abs(local_ang_vel[:, 1])
    )
    return cost

def reward_orientation(
    pelvis_rpy: torch.Tensor, 
    torso_rpy: torch.Tensor, 
    idle_mask: torch.Tensor
) -> torch.Tensor:
    """Reward for maintaining upright orientation.
    
    Args:
        pelvis_rpy: Pelvis RPY angles [num_envs, 3]
        torso_rpy: Torso RPY angles [num_envs, 3]
        idle_mask: Mask for idle state (no motion) [num_envs]
    """
    err_roll = torch.abs(pelvis_rpy[:, 0]) + torch.abs(torso_rpy[:, 0])
    err_pitch_dire = torch.abs(torch.clamp(torso_rpy[:, 1], -np.pi, 0.0))
    err_pitch_idle = idle_mask * torch.abs(torso_rpy[:, 1])
    err_ori = err_roll + err_pitch_dire + err_pitch_idle
    rew = torch.exp(-0.5 * err_ori) - err_pitch_dire
    return rew

def reward_body_rotation(
    legs_navi_rot: torch.Tensor, 
    cmd_yaw_vel: torch.Tensor, 
    cmd_yaw_max: float
) -> torch.Tensor:
    """Reward for aligning legs with the navigation frame.
    
    Args:
        legs_navi_rot: Rotation matrix from legs to navi frame [num_envs, N_legs, 3, 3]
        cmd_yaw_vel: Yaw velocity command [num_envs]
        cmd_yaw_max: Maximum yaw velocity for normalization
    """
    cmd_decay = torch.clamp((cmd_yaw_max - torch.abs(cmd_yaw_vel)) / (cmd_yaw_max + 1e-6), 0.0, 1.0) ** 2
    # legs2navi_rot[:, :, 2, 1] is the (z, y) element, representing roll error
    axis_roll_err = torch.mean(torch.abs(legs_navi_rot[:, :, 2, 1]), dim=1)
    # legs2navi_rot[:, :, 0, 1] is the (x, y) element, representing yaw error
    axis_yaw_err = torch.mean(cmd_decay.unsqueeze(-1) * torch.abs(legs_navi_rot[:, :, 0, 1]), dim=1)
    return torch.exp(-5.0 * (axis_roll_err + axis_yaw_err))

def cost_foot_contact(
    feet_contact: torch.Tensor, 
    gait_mask: torch.Tensor, 
    move_flag: torch.Tensor
) -> torch.Tensor:
    """Penalty for incorrect foot contact according to gait.
    
    Args:
        feet_contact: Binary contact state [num_envs, num_feet]
        gait_mask: Target gait mask [num_envs, num_feet] (1: stance, -1: swing)
        move_flag: Mask for moving state [num_envs]
    """
    stance = (feet_contact > 0).float()
    swing = (feet_contact == 0).float()
    stance_des = (gait_mask == 1).float()
    swing_des = (gait_mask == -1).float()
    is_constrained = (gait_mask != 0).float()
    
    cost_stance = torch.sum(torch.abs(stance - stance_des) * is_constrained, dim=-1)
    cost_swing = torch.sum(torch.abs(swing - swing_des) * is_constrained, dim=-1)
    return (cost_stance + cost_swing) * move_flag

def cost_foot_clearance(
    feet_z: torch.Tensor, 
    target_foot_height: torch.Tensor, 
    gait_mask: torch.Tensor, 
    move_flag: torch.Tensor,
    stance_height: float = 0.0
) -> torch.Tensor:
    """Penalty for incorrect foot height during swing.
    
    Args:
        feet_z: Current feet z-positions [num_envs, num_feet]
        target_foot_height: Desired swing height [num_envs]
        gait_mask: Target gait mask [num_envs, num_feet] (-1: swing)
        move_flag: Mask for moving state [num_envs]
    """
    swing_des = (gait_mask == -1).float()
    foot_z_tar = stance_height + target_foot_height.unsqueeze(-1)
    cost = torch.sum(swing_des * torch.square(feet_z - foot_z_tar), dim=-1)
    return cost * move_flag

def cost_foot_slip(
    feet_vel: torch.Tensor, 
    gait_mask: torch.Tensor
) -> torch.Tensor:
    """Penalty for foot slipping during stance.
    
    Args:
        feet_vel: Feet linear velocities [num_envs, num_feet, 3]
        gait_mask: Target gait mask [num_envs, num_feet] (1: stance)
    """
    stance_des = (gait_mask == 1).float()
    vel_norm_sq = torch.sum(torch.square(feet_vel), dim=-1)
    return torch.sum(vel_norm_sq * stance_des, dim=-1)

def cost_foot_balance(
    com_pos_navi: torch.Tensor, 
    feet_pos_navi: torch.Tensor, 
    move_flag: torch.Tensor
) -> torch.Tensor:
    """Penalty for center-of-mass being far from the support polygon center.
    
    Args:
        com_pos_navi: COM position in navi frame [num_envs, 3]
        feet_pos_navi: Feet positions in navi frame [num_envs, num_feet, 3]
        move_flag: Mask for moving state [num_envs]
    """
    foot2com_err = feet_pos_navi[:, :, :2] - com_pos_navi.unsqueeze(1)[:, :, :2]
    foot_center = foot2com_err[:, 0, :] + foot2com_err[:, 1, :]
    cost_support = torch.sum(torch.square(foot_center), dim=-1)
    
    foot_dist = torch.linalg.norm(feet_pos_navi[:, 0, :2] - feet_pos_navi[:, 1, :2], dim=-1)
    foot_spread_penalty = torch.clamp(0.35 - foot_dist, min=0.0) * 10.0
    
    return cost_support * (1.0 + foot_spread_penalty)

def cost_straight_knee(knee_joint_pos: torch.Tensor) -> torch.Tensor:
    """Penalty for straight knees to avoid singularities.
    
    Args:
        knee_joint_pos: Knee joint positions [num_envs, num_knees]
    """
    penalty = torch.clamp(0.1 - knee_joint_pos, min=0.0)
    return torch.sum(penalty, dim=-1)

def cost_foot_far(feet_pos: torch.Tensor) -> torch.Tensor:
    """Penalty for feet being too close to each other.
    
    Args:
        feet_pos: Feet positions [num_envs, num_feet, 3]
    """
    foot_dist = torch.linalg.norm(feet_pos[:, 0, :] - feet_pos[:, 1, :], dim=-1)
    return torch.clamp(0.35 - foot_dist, min=0.0)

def cost_smoothness_action(
    action: torch.Tensor, 
    last_action: torch.Tensor, 
    last_last_action: torch.Tensor
) -> torch.Tensor:
    """Penalty for high-frequency action changes.
    """
    smooth_0th = torch.square(action) # From cat env
    smooth_1st = torch.square(action - last_action)
    smooth_2nd = torch.square(action - 2 * last_action + last_last_action)
    return torch.sum(smooth_0th + smooth_1st + smooth_2nd, dim=-1)

def cost_smoothness_joint(
    joint_vel: torch.Tensor, 
    last_joint_vel: torch.Tensor, 
    dt: float
) -> torch.Tensor:
    """Penalty for joint velocity and acceleration.
    """
    joint_acc = (joint_vel - last_joint_vel) / dt
    return torch.sum(0.01 * torch.square(joint_vel) + torch.square(joint_acc), dim=-1)

def cost_torque(torques: torch.Tensor) -> torch.Tensor:
    """Penalty for high torques.
    """
    return torch.sum(torch.square(torques), dim=-1)

def cost_joint_pos_limits(
    joint_pos: torch.Tensor, 
    soft_limits_low: torch.Tensor, 
    soft_limits_high: torch.Tensor
) -> torch.Tensor:
    """Penalty for joints exceeding soft limits.
    """
    out_of_limits = -torch.clamp(joint_pos - soft_limits_low, max=0.0)
    out_of_limits += torch.clamp(joint_pos - soft_limits_high, min=0.0)
    return torch.sum(out_of_limits, dim=-1)

def reward_guidance_field_alignment(
    gf_vel: torch.Tensor, 
    lin_vel: torch.Tensor, 
    sdf: torch.Tensor, 
    crossed: torch.Tensor, 
    tau: float = 0.3
) -> torch.Tensor:
    """Reward for aligning velocity with the guidance field.
    
    Args:
        gf_vel: Guidance field velocity vector [num_envs, N, 3]
        lin_vel: Current body linear velocity [num_envs, N, 3]
        sdf: Signed distance field value [num_envs, N, 1]
        crossed: Mask for points that have crossed the target or are in stop mode [num_envs, N]
        tau: Distance threshold for "near" reward
    """
    eps = 1e-6
    k_window = 40.0
    alpha_align = 5.0

    g_norm = gf_vel / (torch.linalg.norm(gf_vel, dim=-1, keepdim=True) + eps)
    v_norm = lin_vel / (torch.linalg.norm(lin_vel, dim=-1, keepdim=True) + eps)
    cos_align = torch.sum(g_norm * v_norm, dim=-1)

    sdf_flat = sdf.squeeze(-1)
    window = torch.sigmoid(k_window * (tau - sdf_flat))

    reward_near = window * (alpha_align * cos_align)
    reward_near = torch.where(crossed, alpha_align * 0.8, reward_near)

    return torch.mean(reward_near, dim=-1)

def cost_sdf_penalty(sdf: torch.Tensor, sdf_safe: float = 0.05) -> torch.Tensor:
    """Penalty for being too close to or inside obstacles.
    
    Args:
        sdf: Signed distance field value [num_envs, N, 1]
        sdf_safe: Safety margin
    """
    beta_inside = 0.02
    pen_inside_scale = 20.0

    sdf_flat = sdf.squeeze(-1)
    pen_inside = torch.nn.functional.softplus((sdf_safe - sdf_flat) / beta_inside)
    penalty = pen_inside_scale * pen_inside

    return -torch.mean(penalty, dim=-1)
    
