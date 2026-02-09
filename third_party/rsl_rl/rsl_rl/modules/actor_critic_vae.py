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

from .actor_vae import Actor_VAE


class Critic_VAE(nn.Module):
    """Critic network for VAE-based actor-critic architecture.
    
    This is an MLP critic that takes a sampled/mean latent vector from VAE encoder
    and proprioceptive state to estimate value.
    
    Network Architecture:
        Input: latent_vector (vae_latent_dim) + proprioceptive_state (critic_sp_dim)
        Hidden layers: critic_hidden_dims
        Output: value estimate (scalar)
        
    Proprioceptive State Normalization:
        - Only the proprioceptive state (x_p) is normalized via EmpiricalNormalization
        - The latent vector from encoder is NOT normalized (it comes from VAE latent space)
    """
    
    def __init__(
        self,
        vae_latent_dim: int,
        critic_sp_dim: int,
        critic_hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
    ):
        super().__init__()
        
        self.vae_latent_dim = vae_latent_dim
        self.critic_sp_dim = critic_sp_dim
        
        activation_fn = resolve_nn_activation(activation)
        
        # Proprioceptive State Normalizer (only for x_p, not for latent space)
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(critic_sp_dim,))
        
        # Critic network: [latent + proprioceptive_state] -> value
        critic_input_dim = vae_latent_dim + critic_sp_dim
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
            z: Latent vector from VAE encoder of shape (batch_size, vae_latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, critic_sp_dim)
        
        Returns:
            value: Value estimate of shape (batch_size, 1)
        """
        # Normalize only the proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Concatenate latent and normalized proprioceptive state
        return self.critic_mlp(torch.cat([z, x_sp_normalized], dim=-1))


class ActorCriticVAE(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # actor vae specific
        actor_sg_dim: int = None,
        actor_sh_dim: int = 0,  # smplx human state dimension
        vae_latent_dim: int = 32,
        robot_encoder_hidden_dims: list[int] = None,
        human_encoder_hidden_dims: list[int] = None,
        recover_decoder_hidden_dims: list[int] = None,
        activate_signals: Literal["robot", "smplx"] = "robot",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticVAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_fn = resolve_nn_activation(activation)
        
        # Set default hidden dims if not provided
        if robot_encoder_hidden_dims is None:
            robot_encoder_hidden_dims = [512, 256]
        if human_encoder_hidden_dims is None:
            human_encoder_hidden_dims = [512, 256]
        if recover_decoder_hidden_dims is None:
            recover_decoder_hidden_dims = [256, 512]
        
        self.activate_signals = activate_signals
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim

        #####################################################################################
        # Actor: VAE Policy
        # Signal flow:
        #    s^g --encoder--> z^g --recover_decoder--> s^g_reconstructed
        #                      │
        #                      ├──[concat]── action_decoder── action
        #    s^p ──────────────┘
        #####################################################################################
        
        self.actor = Actor_VAE(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_sg_dim=actor_sg_dim,
            actor_sh_dim=actor_sh_dim,
            vae_latent_dim=vae_latent_dim,
            robot_encoder_hidden_dims=robot_encoder_hidden_dims,
            human_encoder_hidden_dims=human_encoder_hidden_dims,
            action_decoder_hidden_dims=actor_hidden_dims,
            recover_decoder_hidden_dims=recover_decoder_hidden_dims,
            activation=activation,
            activate_signals=activate_signals,
        )

        #####################################################################################
        # Critic: Value function
        # Signal flow:
        #     z (from actor encoder) ──┐
        #                              ├──critic_mlp── V(s)
        #     s^p (proprioceptive) ────┘
        #####################################################################################
        
        # Calculate critic proprioceptive dimension
        critic_sp_dim = num_critic_obs - actor_sg_dim - actor_sh_dim
        
        self.critic = Critic_VAE(
            vae_latent_dim=vae_latent_dim,
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
    # not used at the moment
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
        # compute mean
        mean = self.actor(observations)
        
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
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
        # Extract proprioceptive state from critic observations
        x_sp_critic = critic_observations[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Shared Actor Encoder - encode based on activate_signals
        if self.activate_signals == "robot":
            x_sg = critic_observations[:, :self.actor_sg_dim]
            z_sg, _, _ = self.actor.encode_robot(x_sg)
        elif self.activate_signals == "smplx":
            x_sh = critic_observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
            z_sg, _, _ = self.actor.encode_smplx(x_sh)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}")
        
        # Critic MLP: [z, s^p] -> value (Critic_VAE.forward handles normalization)
        value = self.critic(z_sg, x_sp_critic)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_inference_reconstruction(self, observations):
        """Get reconstruction of s^g from VAE for analysis.
        
        Performs reconstruction for both robot and human modalities with sampling.
        
        Args:
            observations: Input observations containing robot state, human state, and proprioception.
        
        Returns:
            Tuple of (original_sg, reconstructed_sg_robot, reconstructed_sg_human)
            where reconstructed_sg_robot and reconstructed_sg_human are reconstructions
            from sampled latent vectors.
        """
        # Get original state
        x_sg = observations[:, :self.actor_sg_dim]
        
        # Encode with sampling for both modalities
        z_robot, mu_robot, logvar_robot = self.actor.encode_robot(observations)
        z_human, mu_human, logvar_human = self.actor.encode_smplx(observations)
        
        # Reconstruct from sampled latent vectors
        recon_sg_robot = self.actor.decode_recover(z_robot)
        recon_sg_human = self.actor.decode_recover(z_human)
        
        return x_sg, recon_sg_robot, recon_sg_human