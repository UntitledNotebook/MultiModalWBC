# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.normalizer import EmpiricalNormalization

from .actor_projection import Actor_Projection


class Critic_Projection(nn.Module):
    """Critic network for projection-based actor-critic architecture.
    
    This is a simple MLP critic that takes a projected latent vector and 
    proprioceptive state to estimate value.
    
    Network Architecture:
        Input: projected_latent (projection_hidden_dims) + proprioceptive_state (critic_sp_dim)
        Hidden layers: critic_hidden_dims
        Output: value estimate (scalar)
    """
    
    def __init__(
        self,
        projection_hidden_dims: int,
        critic_sp_dim: int,
        critic_hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
    ):
        super().__init__()
        
        self.projection_hidden_dims = projection_hidden_dims
        self.critic_sp_dim = critic_sp_dim
        
        activation_fn = resolve_nn_activation(activation)
        
        # Proprioceptive State Normalizer
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(critic_sp_dim,))
        
        # Critic network: [projected_latent + proprioceptive_state] -> value
        critic_input_dim = projection_hidden_dims + critic_sp_dim
        critic_layers = []
        
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i+1]))
            critic_layers.append(activation_fn)
        
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        
        self.critic_mlp = nn.Sequential(*critic_layers)
    
    def forward(self, z, x_sp):
        """Forward pass of the critic.
        
        Args:
            z: Projected latent vector of shape (batch_size, projection_hidden_dims)
            x_sp: Proprioceptive state of shape (batch_size, critic_sp_dim)
        
        Returns:
            value: Value estimate of shape (batch_size, 1)
        """
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        return self.critic_mlp(torch.cat([z, x_sp_normalized], dim=-1))


class ActorCriticProjection(nn.Module):
    """Actor-Critic network with projection-based multi-modal alignment.
    
    This module implements an actor-critic architecture where the actor uses
    projection networks to map robot and human command observations to a shared
    latent space and decode actions, while the critic estimates value from the
    same projected latent representation.
    
    The activate_signals parameter controls which modality (robot or smplx) is
    used during inference, affecting both actor and critic forward passes.
    
    Architecture:
        Actor:
            - Robot Projection Network: state_goal -> projection_hidden_dims
            - Human Projection Network: smplx_human_state -> projection_hidden_dims
            - Action Decoder: projected_latent + proprioceptive_state -> actions
            - Recover Decoder: NOT USED (projection-based, no reconstruction)
        
        Critic:
            - MLP: Estimates value from projected_latent + proprioceptive_state
        
    Observation Split:
        Actor Observations:
            - Robot state: First `actor_sg_dim` dimensions
            - Human state: Next `actor_sh_dim` dimensions
            - Proprioceptive state: Remaining dimensions
        
        Critic Observations:
            - Robot state: First `actor_sg_dim` dimensions
            - Proprioceptive state: Remaining dimensions (no human state)
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        # Projection-specific parameters
        actor_sg_dim: int = None,
        actor_sh_dim: int = None,
        projection_hidden_dims: int = 64,
        activate_signals: Literal["robot", "smplx"] = "robot",
        robot_projection_hidden_dims: list[int] = None,
        human_projection_hidden_dims: list[int] = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticProjection.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        if robot_projection_hidden_dims is None:
            robot_projection_hidden_dims = [512, 256]
        if human_projection_hidden_dims is None:
            human_projection_hidden_dims = [512, 256]
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.activate_signals = activate_signals
        
        #####################################################################################
        # Actor: Projection-based Policy
        # Signal flow:
        #     state_goal ──robot_projection── latent ──┐
        #                                               ├──[concat with prop_state]──action_decoder── action
        #     proprioceptive_state ──normalizer────────┤
        #                                               │
        #     smplx_state ──human_projection── latent ──┘
        #####################################################################################
        
        self.actor = Actor_Projection(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_sg_dim=actor_sg_dim,
            actor_sh_dim=actor_sh_dim,
            activate_signals=activate_signals,
            robot_projection_hidden_dims=robot_projection_hidden_dims,
            human_projection_hidden_dims=human_projection_hidden_dims,
            projection_hidden_dims=projection_hidden_dims,
            action_hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        #####################################################################################
        # Critic: Value function
        # Signal flow:
        #     z (from actor projection) ──┐
        #                                 ├──critic_mlp── V(s)
        #     s^p (proprioceptive) ──────┘
        #####################################################################################
        
        # Calculate critic proprioceptive dimension
        # Critic uses only actor_sg_dim + actor_sh_dim  + proprioceptive_state
        critic_sp_dim = num_critic_obs - actor_sg_dim - actor_sh_dim
        
        self.critic = Critic_Projection(
            projection_hidden_dims=projection_hidden_dims,
            critic_sp_dim=critic_sp_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """Compute mean from projection-based actor and create distribution.
        
        Args:
            observations: Input observations containing robot state, human state, and proprioception.
                         Shape: (batch_size, num_actor_obs)
        """
        # Forward pass through projection actor
        # The actor automatically routes based on activate_signals
        mean = self.actor(observations)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        """Sample action from the learned distribution.
        
        Args:
            observations: Input observations.
        
        Returns:
            Sampled actions.
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Get log probability of actions.
        
        Args:
            actions: Actions to evaluate.
        
        Returns:
            Log probability of actions.
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """Get deterministic action for inference.
        
        Args:
            observations: Input observations.
        
        Returns:
            Predicted actions (mean of the distribution).
        """
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value from critic using activate_signals to choose modality.
        
        Args:
            critic_observations: Observations for the critic.
                                Shape: (batch_size, num_critic_obs)
                                Structure: [state_goal | proprioceptive_state]
        
        Returns:
            Value estimates from the critic.
        """
        # Project based on activate_signals
        if self.activate_signals == "robot":
            # Use robot projection
            z = self.actor.robot_projection(critic_observations[:, :self.actor_sg_dim])
        elif self.activate_signals == "smplx":
            # Use human projection
            z = self.actor.human_projection(critic_observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim])
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}. Must be 'robot' or 'smplx'.")
        
        # Extract proprioceptive state from critic observations
        # Critic observations structure: [state_goal | proprioceptive_state]
        x_sp = critic_observations[:, self.actor_sg_dim+self.actor_sh_dim:]
        
        # Critic: [z, s^p] -> value
        # (Note: x_sp will be normalized inside critic.forward)
        value = self.critic(z, x_sp)
        
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic-projection model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training.
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

    ###################################################
    #        PROJECTION-BASED AUXILIARY OUTPUTS       #
    ###################################################
    
    def act_inference_robot(self, observations):
        """Get deterministic action for robot state only.
        
        Args:
            observations: Input observations.
        
        Returns:
            Predicted actions using robot projection.
        """
        actions_mean = self.actor.forward_robot(observations)
        return actions_mean
    
    def act_inference_smplx(self, observations):
        """Get deterministic action for SMPLX human state only.
        
        Args:
            observations: Input observations.
        
        Returns:
            Predicted actions using human projection.
        """
        actions_mean = self.actor.forward_smplx(observations)
        return actions_mean

    def get_projections(self, observations):
        """Get both robot and human projections for alignment loss computation.
        
        Args:
            observations: Input observations.
        
        Returns:
            Tuple of (z_robot, z_human) projected latent vectors.
        """
        z_robot = self.actor.project_robot(observations)
        z_human = self.actor.project_smplx(observations)
        return z_robot, z_human


