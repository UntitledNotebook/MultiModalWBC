"""
VAE Autoencoder with dual decoders for robot control and recovery.

This module implements a Variational Autoencoder (VAE) architecture designed for
robot motion control. It encodes goal state to a latent representation and decodes
it for both action generation and state reconstruction.

Network Architecture:
    Encoder:
        - Input: state_goal (actor_sg_dim) 
        - Hidden layers: robot_encoder_hidden_dims
        - Output: μ, log(σ²) (latent_dim each)
    
    Robot Decoder (Action Decoder):
        - Input: latent vector (latent_dim) + proprioceptive state (actor_sp_dim)
        - Hidden layers: action_decoder_hidden_dims
        - Output: action (num_actions)
    
    Recover Decoder (State Reconstruction):
        - Input: latent vector (latent_dim)
        - Hidden layers: recover_decoder_hidden_dims
        - Output: reconstructed state_goal (actor_sg_dim)

Signal flow:
    state_goal ──encoder── latent ──recover_decoder── reconstructed_state_goal
                             │
                             ├──[concat with prop_state]──action_decoder── action
                             │          
    prop_state ──────────────┘   
"""

import torch
import torch.nn as nn
from typing import Literal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.normalizer import EmpiricalNormalization

class Actor_VAE(nn.Module):
    def __init__(
        self,
        # In-Out Dimensions
        num_actor_obs: int,
        num_actions: int,
        actor_sg_dim: int,
        actor_sh_dim: int = 0,  # smplx human state dimension
        # VAE Configuration
        vae_latent_dim: int = 32,
        # Network Architecture
        robot_encoder_hidden_dims: list[int] = [512, 256],
        human_encoder_hidden_dims: list[int] = [512, 256],
        action_decoder_hidden_dims: list[int] = [256, 128],
        recover_decoder_hidden_dims: list[int] = [256, 512],
        activation: str = "relu",
        # Modality routing
        activate_signals: Literal["robot", "smplx"] = "robot",
    ):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.num_actions = num_actions
        self.activate_signals = activate_signals
        self.vae_latent_dim = vae_latent_dim

        activation_fn = resolve_nn_activation(activation)

        # Robot Encoder: state_goal -> Latent (mean and logvar)
        robot_encoder_layers = []
        robot_encoder_layers.append(nn.Linear(actor_sg_dim, robot_encoder_hidden_dims[0]))
        robot_encoder_layers.append(activation_fn)
        for i in range(len(robot_encoder_hidden_dims) - 1):
            robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[i], robot_encoder_hidden_dims[i+1]))
            robot_encoder_layers.append(activation_fn)
        self.robot_encoder = nn.Sequential(*robot_encoder_layers)
        
        # Human Encoder: smplx_human_state -> Latent (mean and logvar)
        human_encoder_layers = []
        human_encoder_layers.append(nn.Linear(actor_sh_dim, human_encoder_hidden_dims[0]))
        human_encoder_layers.append(activation_fn)
        for i in range(len(human_encoder_hidden_dims) - 1):
            human_encoder_layers.append(nn.Linear(human_encoder_hidden_dims[i], human_encoder_hidden_dims[i+1]))
            human_encoder_layers.append(activation_fn)
        self.human_encoder = nn.Sequential(*human_encoder_layers)
        
        # Shared μ and log(σ²) heads for both encoders
        self.fc_mu = nn.Linear(robot_encoder_hidden_dims[-1], vae_latent_dim)
        self.fc_logvar = nn.Linear(robot_encoder_hidden_dims[-1], vae_latent_dim)

        # Proprioceptive State Normalizer (only for x_p, not x_sg or x_sh)
        actor_sp_dim = num_actor_obs - actor_sg_dim - actor_sh_dim
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(actor_sp_dim,))
        
        # Action Decoder: Latent + Proprioceptive State -> Actions
        action_layers = []
        action_layers.append(nn.Linear(vae_latent_dim + actor_sp_dim, action_decoder_hidden_dims[0]))
        action_layers.append(activation_fn)
        for i in range(len(action_decoder_hidden_dims) - 1):
            action_layers.append(nn.Linear(action_decoder_hidden_dims[i], action_decoder_hidden_dims[i+1]))
            action_layers.append(activation_fn)
        action_layers.append(nn.Linear(action_decoder_hidden_dims[-1], num_actions))
        self.action_decoder = nn.Sequential(*action_layers)

        # Recover Decoder: Latent -> Reconstruction of state_goal
        recover_layers = []
        recover_layers.append(nn.Linear(vae_latent_dim, recover_decoder_hidden_dims[0]))
        recover_layers.append(activation_fn)
        for i in range(len(recover_decoder_hidden_dims) - 1):
            recover_layers.append(nn.Linear(recover_decoder_hidden_dims[i], recover_decoder_hidden_dims[i+1]))
            recover_layers.append(activation_fn)
        recover_layers.append(nn.Linear(recover_decoder_hidden_dims[-1], actor_sg_dim))
        self.recover_decoder = nn.Sequential(*recover_layers)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through the VAE actor with modality routing.
        
        During inference (forward), uses μ (mean) without sampling to ensure
        deterministic action generation. The reparameterization sampling is
        only used during training via encode_with_sampling().
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        if self.activate_signals == "robot":
            return self.forward_robot(x)
        elif self.activate_signals == "smplx":
            return self.forward_smplx(x)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}. Must be 'robot' or 'smplx'.")
    
    def forward_robot(self, x):
        """Forward pass using robot goal state modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input into sg (robot state) and sp (proprioceptive state)
        x_sg = x[:, :self.actor_sg_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Encode sg to latent
        encoded = self.robot_encoder(x_sg)
        mu = self.fc_mu(encoded)
        
        # Use μ (mean) for inference, not reparameterization
        z_sg = mu
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sg, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_smplx(self, x):
        """Forward pass using SMPLX human state modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input into sh (human state) and sp (proprioceptive state)
        x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Encode sh to latent
        encoded = self.human_encoder(x_sh)
        mu = self.fc_mu(encoded)
        
        # Use μ (mean) for inference, not reparameterization
        z_sh = mu
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sh, x_sp_normalized], dim=-1))
        
        return actions

    def encode_with_sampling(self, x):
        """Encode with reparameterization sampling for training.
        
        This method is used during training to get z, μ, and log(σ²) for
        computing reconstruction loss and KL divergence.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            z: Sampled latent vector of shape (batch_size, vae_latent_dim)
            mu: Mean of latent distribution of shape (batch_size, vae_latent_dim)
            logvar: Log variance of latent distribution of shape (batch_size, vae_latent_dim)
        """
        if self.activate_signals == "robot":
            x_sg = x[:, :self.actor_sg_dim]
            encoded = self.robot_encoder(x_sg)
        elif self.activate_signals == "smplx":
            x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
            encoded = self.human_encoder(x_sh)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}")
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)  # Sampling for training
        
        return z, mu, logvar
    
    def encode_robot(self, x):
        """Encode using robot encoder with sampling for training.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs) or (batch_size, actor_sg_dim)
        
        Returns:
            z: Sampled latent vector
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sg = x[:, :self.actor_sg_dim]
        else:
            x_sg = x
        
        encoded = self.robot_encoder(x_sg)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def encode_smplx(self, x):
        """Encode using human encoder with sampling for training.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs) or (batch_size, actor_sh_dim)
        
        Returns:
            z: Sampled latent vector
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        else:
            x_sh = x
        
        encoded = self.human_encoder(x_sh)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

    def decode_action(self, z, x_sp):
        """Decode actions from latent and proprioceptive state.
        
        Args:
            z: Latent vector of shape (batch_size, vae_latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        return self.action_decoder(torch.cat([z, x_sp_normalized], dim=-1))

    def decode_recover(self, z):
        """Decode reconstruction from latent.
        
        Args:
            z: Latent vector of shape (batch_size, vae_latent_dim)
        
        Returns:
            recon: Reconstructed state_goal of shape (batch_size, actor_sg_dim)
        """
        return self.recover_decoder(z)
    
    def forward_robot_exporter(self, robot_command, proprioceptive_state):
        """Forward pass for robot command and proprioceptive state.
        
        This method supports decoupled inputs for easier deployment.
        
        Args:
            robot_command: Input tensor of shape (batch_size, actor_sg_dim)
            proprioceptive_state: Input tensor of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Encode robot state to latent
        encoded = self.robot_encoder(robot_command)
        mu = self.fc_mu(encoded)
        z_sg = mu  # Use mean for inference
        
        # Normalize proprioceptive state
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sg, proprioceptive_state_normalized], dim=-1))
        
        return actions
    
    def forward_smplx_exporter(self, smplx_human_state, proprioceptive_state):
        """Forward pass for SMPLX human state and proprioceptive state.
        
        This method supports decoupled inputs for easier deployment.
        
        Args:
            smplx_human_state: Input tensor of shape (batch_size, actor_sh_dim)
            proprioceptive_state: Input tensor of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Encode human state to latent
        encoded = self.human_encoder(smplx_human_state)
        mu = self.fc_mu(encoded)
        z_sh = mu  # Use mean for inference
        
        # Normalize proprioceptive state
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sh, proprioceptive_state_normalized], dim=-1))
        
        return actions

    def train(self, mode: bool = True):
        """Override train() to ensure frozen modules stay in eval mode during finetune.
        
        The proprioceptive_normalizer is kept in training mode to continuously
        update its normalization statistics.
        """
        super().train(mode)
        # Ensure frozen modules remain in eval mode if mode=True
        if mode and hasattr(self, '_frozen_for_finetune') and self._frozen_for_finetune:
            frozen_modules: list[nn.Module] = [
                self.robot_encoder,
                self.action_decoder,
                self.recover_decoder,
                self.proprioceptive_normalizer,
                self.fc_mu,
                self.fc_logvar,
            ]
            for module in frozen_modules:
                module.eval()
    
    def freeze_for_finetune(self):
        """Freeze all modules except human encoder for finetuning on SMPLX modality.
        
        This method:
        1. Freezes robot encoder, action decoder, recover decoder, shared projection heads (fc_mu, fc_logvar)
        2. Enables gradient computation only for human encoder
        3. Goal: optimize human encoder to better align with shared latent space
        
        Call this before finetuning with smplx signals.
        """
        self._frozen_for_finetune = True
        
        # Freeze all modules except human_encoder
        frozen_modules: list[nn.Module] = [
            self.robot_encoder,
            self.action_decoder,
            self.recover_decoder,
            self.proprioceptive_normalizer,
            self.fc_mu,
            self.fc_logvar,
        ]
        for module in frozen_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        
        # Only human_encoder is trainable for finetune
        self.human_encoder.train()
        for param in self.human_encoder.parameters():
            param.requires_grad = True