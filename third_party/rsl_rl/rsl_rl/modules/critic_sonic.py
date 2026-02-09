"""
Critic module for Actor-Critic with FSQ-VAE based architecture.

This module implements a value function (critic) that works with the FSQ-VAE actor
to estimate state values for policy gradient algorithms.

Network Architecture:
    Input:
        - Quantized latent vector from robot encoder (fsqvae_latent_dim)
        - Proprioceptive state (actor_sp_dim)
    
    MLP:
        - Hidden layers: critic_hidden_dims
        - Output: value estimate (1)

Signal flow:
    state_goal ──robot_encoder── latent ──fsq── z_q ──┐
                                                       ├──[concat with prop_state]──critic_mlp── value
    prop_state ──────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class Critic_SONIC(nn.Module):
    """Value function for FSQ-VAE based actor-critic architecture.
    
    The critic estimates state values from:
    1. Quantized latent representation from the robot encoder
    2. Proprioceptive state observations
    
    Both inputs are normalized for stable training.
    """
    
    def __init__(
        self,
        fsqvae_latent_dim: int,
        critic_sp_dim: int,
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
    ):
        """Initialize Critic_SONIC module.
        
        Args:
            fsqvae_latent_dim: Dimension of quantized latent from FSQ-VAE
            critic_sp_dim: Dimension of proprioceptive state
            critic_hidden_dims: Hidden layer dimensions for MLP
            activation: Activation function type (e.g., "elu", "relu")
        """
        super().__init__()
        
        self.fsqvae_latent_dim = fsqvae_latent_dim
        self.critic_sp_dim = critic_sp_dim
        
        activation_fn = resolve_nn_activation(activation)
        
        # Normalizers for inputs
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(critic_sp_dim,))
        
        # MLP for value estimation: [z_q, s^p] -> V(s)
        mlp_input_dim = fsqvae_latent_dim + critic_sp_dim
        
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation_fn)
        
        self.mlp = nn.Sequential(*critic_layers)
    
    def forward(self, z_q, x_sp):
        """Compute value estimate from quantized latent and proprioceptive state.
        
        Args:
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, actor_sp_dim)
        
        Returns:
            value: Value estimate of shape (batch_size, 1)
        """
        # Normalize inputs
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Concatenate and pass through MLP
        combined = torch.cat([z_q, x_sp_normalized], dim=-1)
        value = self.mlp(combined)
        
        return value