"""
FSQ-VAE Autoencoder with dual decoders for robot control and recovery.

This module implements a Finite Scalar Quantization Variational Autoencoder (FSQ-VAE)
architecture designed for robot motion control. It encodes goal state to a quantized
latent representation and decodes it for both action generation and state reconstruction.

Network Architecture:
    Encoder:
        - Input: state_goal (actor_sg_dim) 
        - Hidden layers: robot_encoder_hidden_dims
        - Output: latent vector (fsqvae_latent_dim)
    
    FSQ Quantizer:
        - Input: latent vector (fsqvae_latent_dim)
        - Quantization: Finite Scalar Quantization
        - Output: quantized latent vector, quantization indices
    
    Action Decoder:
        - Input: quantized latent vector (fsqvae_latent_dim) + proprioceptive state (actor_sp_dim)
        - Hidden layers: action_decoder_hidden_dims
        - Output: action (num_actions)
    
    Recover Decoder (State Reconstruction):
        - Input: quantized latent vector (fsqvae_latent_dim)
        - Hidden layers: recover_decoder_hidden_dims
        - Output: reconstructed state_goal (actor_sg_dim)

Signal flow:
    state_goal ──encoder── latent ──fsq_quantizer── z_quantized ──recover_decoder── reconstructed_state_goal
                                                          │
                                                          ├──[concat with prop_state]──action_decoder── action
                                                          │          
    prop_state ───────────────────────────────────────────┘   
"""

import torch
import torch.nn as nn
from rsl_rl.modules.finite_scalar_quantization import FSQ
from rsl_rl.utils import resolve_nn_activation

class Actor_FSQVAE(nn.Module):
    def __init__(
        self,
        # In-Out Dimensions
        num_actor_obs: int,
        num_actions: int,
        actor_sg_dim: int,
        # FSQ-VAE Configuration
        fsqvae_latent_dim: int,
        levels: list[int],
        num_codebooks: int = 1,
        # Network Architecture
        robot_encoder_hidden_dims: list[int] = [512, 256],
        action_decoder_hidden_dims: list[int] = [256, 128],
        recover_decoder_hidden_dims: list[int] = [256, 512],
        activation: str = "relu",
    ):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.actor_sg_dim = actor_sg_dim
        self.num_actions = num_actions
        self.fsqvae_latent_dim = fsqvae_latent_dim

        activation_fn = resolve_nn_activation(activation)

        # Encoder: state_goal -> Latent
        robot_encoder_layers = []
        robot_encoder_layers.append(nn.Linear(actor_sg_dim, robot_encoder_hidden_dims[0]))
        robot_encoder_layers.append(activation_fn)
        for i in range(len(robot_encoder_hidden_dims) - 1):
            robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[i], robot_encoder_hidden_dims[i+1]))
            robot_encoder_layers.append(activation_fn)
        robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[-1], fsqvae_latent_dim))
        self.robot_encoder = nn.Sequential(*robot_encoder_layers)

        # FSQ Quantizer: Latent -> Quantized Latent
        self.fsq = FSQ(levels=levels, dim=fsqvae_latent_dim, num_codebooks=num_codebooks)

        # Action Decoder: Quantized Latent + Proprioceptive State -> Actions
        action_layers = []
        action_layers.append(nn.Linear(fsqvae_latent_dim + self.num_actor_obs - self.actor_sg_dim, action_decoder_hidden_dims[0]))
        action_layers.append(activation_fn)
        for i in range(len(action_decoder_hidden_dims) - 1):
            action_layers.append(nn.Linear(action_decoder_hidden_dims[i], action_decoder_hidden_dims[i+1]))
            action_layers.append(activation_fn)
        action_layers.append(nn.Linear(action_decoder_hidden_dims[-1], num_actions))
        self.action_decoder = nn.Sequential(*action_layers)

        # Recover Decoder: Quantized Latent -> Reconstruction of state_goal
        recover_layers = []
        recover_layers.append(nn.Linear(fsqvae_latent_dim, recover_decoder_hidden_dims[0]))
        recover_layers.append(activation_fn)
        for i in range(len(recover_decoder_hidden_dims) - 1):
            recover_layers.append(nn.Linear(recover_decoder_hidden_dims[i], recover_decoder_hidden_dims[i+1]))
            recover_layers.append(activation_fn)
        recover_layers.append(nn.Linear(recover_decoder_hidden_dims[-1], actor_sg_dim))
        self.recover_decoder = nn.Sequential(*recover_layers)

    # def forward(self, x):
    #     """Forward pass through the FSQ-VAE.
        
    #     Args:
    #         x: Input tensor of shape (batch_size, num_actor_obs)
    #            where first actor_sg_dim elements are goal state,
    #            and remaining elements are proprioceptive state.
        
    #     Returns:
    #         actions: Predicted actions of shape (batch_size, num_actions)
    #         recon: Reconstructed goal state of shape (batch_size, actor_sg_dim)
    #         z: Pre-quantization latent vector of shape (batch_size, fsqvae_latent_dim)
    #         z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
    #         indices: Quantization indices from FSQ
    #     """
    #     # Split input into sg and sp
    #     x_sg = x[:, :self.actor_sg_dim]
    #     x_sp = x[:, self.actor_sg_dim:]

    #     # Encode sg to latent
    #     z = self.robot_encoder(x_sg)
        
    #     # Quantize latent
    #     z_q, indices = self.fsq(z)

    #     # Decode actions
    #     actions = self.action_decoder(torch.cat([z_q, x_sp], dim=-1))

    #     # Decode reconstruction
    #     recon = self.recover_decoder(z_q)

    #     return actions, recon, z, z_q, indices
    
    def forward(self, x):
        """Forward pass through the FSQ-VAE.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
               where first actor_sg_dim elements are goal state,
               and remaining elements are proprioceptive state.
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
            recon: Reconstructed goal state of shape (batch_size, actor_sg_dim)
            z: Pre-quantization latent vector of shape (batch_size, fsqvae_latent_dim)
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
            indices: Quantization indices from FSQ
        """
        # Split input into sg and sp
        x_sg = x[:, :self.actor_sg_dim]
        x_sp = x[:, self.actor_sg_dim:]

        # Encode sg to latent
        z = self.robot_encoder(x_sg)
        
        # Quantize latent
        z_q, indices = self.fsq(z)

        # Decode actions
        actions = self.action_decoder(torch.cat([z_q, x_sp], dim=-1))

        return actions

    def encode(self, x):
        """Encode input sg to quantized latent space.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
            indices: Quantization indices from FSQ
        """
        x_sg = x[:, :self.actor_sg_dim]
        z = self.robot_encoder(x_sg)
        z_q, indices = self.fsq(z)
        return z_q, indices

    def decode_action(self, z_q, x_sp):
        """Decode actions from quantized latent and proprioceptive state.
        
        Args:
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, num_actor_obs - actor_sg_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        return self.action_decoder(torch.cat([z_q, x_sp], dim=-1))

    def decode_recover(self, z_q):
        """Decode reconstruction of goal state from quantized latent.
        
        Args:
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
        
        Returns:
            recon: Reconstructed goal state of shape (batch_size, actor_sg_dim)
        """
        return self.recover_decoder(z_q)