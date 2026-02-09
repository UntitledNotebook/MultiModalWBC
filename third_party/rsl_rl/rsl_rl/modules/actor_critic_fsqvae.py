# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

from .actor_fsqvae import Actor_FSQVAE


class ActorCriticFSQVAE(nn.Module):
    """Actor-Critic network with FSQ-VAE-based actor and standard MLP critic.
    
    This module implements an actor-critic architecture where the actor uses
    a Finite Scalar Quantization VAE (FSQ-VAE) to encode command observations
    and decode both actions and reconstructed commands, while the critic uses
    the shared encoder's quantized latent representation.
    
    Architecture:
        Actor:
            s^g (command) ──encoder── z^g ──fsq── z^g_q ──action_decoder── action
                                                        └──recover_decoder── s^g_recon
            s^p (proprioception) ─────────────────┘
        
        Critic:
            s^g (command) ──encoder── z^g_q ──[concat with s^p]── critic.forward ── V(s)
            s^p (proprioception) ────────────────┘
    """
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
        # FSQ-VAE specific
        actor_sg_dim: int = None,
        fsqvae_latent_dim: int = 32,
        fsq_levels: list[int] = None,
        num_codebooks: int = 1,
        robot_encoder_hidden_dims: list[int] = None,
        recover_decoder_hidden_dims: list[int] = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticFSQVAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_fn = resolve_nn_activation(activation)
        
        #####################################################################################
        # Actor: FSQ-VAE Policy
        # Signal flow:
        #    s^g --encoder--> z^g --fsq--> z^g_q --recover_decoder--> s^g_reconstructed
        #                                   │
        #                                   ├──[concat]── action_decoder── action
        #    s^p ───────────────────────────┘
        #####################################################################################
        
        self.actor = Actor_FSQVAE(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_sg_dim=actor_sg_dim,
            fsqvae_latent_dim=fsqvae_latent_dim,
            levels=fsq_levels,
            num_codebooks=num_codebooks,
            robot_encoder_hidden_dims=robot_encoder_hidden_dims,
            action_decoder_hidden_dims=actor_hidden_dims,
            recover_decoder_hidden_dims=recover_decoder_hidden_dims,
            activation=activation,
        )

        #####################################################################################
        # Critic: Value function
        # Signal flow:
        #     s^g --actor.encoder--> z^g --fsq--> z^g_q 
        #                                          │
        #                                          ├──[concat]── critic.forward ── V(s)          
        #     s^p ─────────────────────────────────┘
        #####################################################################################
        
        mlp_input_dim_c = num_critic_obs - actor_sg_dim + fsqvae_latent_dim
        
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

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
        """Compute mean from FSQ-VAE actor and create distribution.
        
        Args:
            observations: Input observations containing both command and proprioception.
        """
        # Forward pass through FSQ-VAE actor
        mean = self.actor(observations)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        """Sample action from the learned distribution."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Get log probability of actions."""
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
        """Evaluate value from critic.
        
        Args:
            critic_observations: Observations for the critic (typically same as actor_observations).
        
        Returns:
            Value estimates from the critic.
        """
        # Split observations state proprioception
        x_sp_critic = critic_observations[:, self.actor.actor_sg_dim:]
        
        # Shared Actor Encoder
        z_q, indices = self.actor.encode(critic_observations)
        
        # Critic MLP: [z_q, s^p] -> value
        value = self.critic(torch.cat([z_q, x_sp_critic], dim=-1))
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic-autoencoder model.

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
    #        FSQ-VAE AUXILIARY OUTPUTS FOR LOSS       #
    ###################################################

    def get_inference_reconstruction(self, observations):
        """Get reconstruction of s^g from FSQ-VAE for analysis.
        
        Args:
            observations: Input observations.
        
        Returns:
            Tuple of (original_sg, reconstructed_sg) for loss computation.
        """
        x_sg = observations[:, :self.actor.actor_sg_dim]
        z_q, indices = self.actor.encode(observations)
        recon = self.actor.decode_recover(z_q)
        return x_sg, recon