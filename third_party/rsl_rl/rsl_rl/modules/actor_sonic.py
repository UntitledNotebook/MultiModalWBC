"""
FSQ-VAE-V2 Autoencoder with dual decoders for robot control and recovery.

This module implements a Finite Scalar Quantization Variational Autoencoder (FSQ-VAE)
architecture designed for robot motion control. It encodes goal state to a quantized
latent representation and decodes it for both action generation and state reconstruction.

Network Architecture:
    Robot Encoder:
        - Input: state_goal (actor_sg_dim) 
        - Hidden layers: robot_encoder_hidden_dims
        - Output: latent vector (fsqvae_latent_dim)
        
    HumanMotion Encoder:
        - Input: smplx_human_state (actor_sh_dim)
        - Hidden layers: human_encoder_hidden_dims
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
    human_state ──human_encoder── latent ──┐       ┌───────zh_quantized ────┐        ┌───  reconstructed_human
                                           │       │            ↓           │        │                   ↓
                                           │       │        loss_token      │        │              loss_recon
                                           │       │            ↑           │        │                   ↑
    state_goal  ──robot_encoder── latent ──fsq_quantizer── zg_quantized ──recover_decoder── reconstructed_state_goal
                                                          │
                                                          ├──[concat with prop_state]──action_decoder── action
                                                          │          
    prop_state ───────────────────────────────────────────┘   
"""
from typing import Literal
import torch
import torch.nn as nn
from rsl_rl.modules.finite_scalar_quantization import FSQ
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.normalizer import EmpiricalNormalization

class Actor_SONIC(nn.Module):
    def __init__(
        self,
        # In-Out Dimensions
        num_actor_obs: int,
        num_actions: int,
        actor_sg_dim: int, # robot goal state dimension
        actor_sh_dim: int, # smplx human state dimension
        # FSQ-VAE Configuration
        fsqvae_latent_dim: int,
        levels: list[int],
        num_codebooks: int = 1,
        # route parameters
        activate_signals: Literal["robot", "smplx"] = "robot",
        # Network Architecture
        robot_encoder_hidden_dims: list[int] = [512, 256],
        human_encoder_hidden_dims: list[int] = [512, 256],
        action_decoder_hidden_dims: list[int] = [256, 128],
        recover_decoder_hidden_dims: list[int] = [256, 512],
        activation: str = "relu",
    ):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.num_actions = num_actions
        self.fsqvae_latent_dim = fsqvae_latent_dim
        
        self.activate_signals = activate_signals

        activation_fn = resolve_nn_activation(activation)

        # Robot Encoder: state_goal -> Latent
        robot_encoder_layers = []
        robot_encoder_layers.append(nn.Linear(actor_sg_dim, robot_encoder_hidden_dims[0]))
        robot_encoder_layers.append(activation_fn)
        for i in range(len(robot_encoder_hidden_dims) - 1):
            robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[i], robot_encoder_hidden_dims[i+1]))
            robot_encoder_layers.append(activation_fn)
        robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[-1], fsqvae_latent_dim))
        self.robot_encoder = nn.Sequential(*robot_encoder_layers)

        # Human Motion Encoder: smplx_human_state -> Latent
        human_encoder_layers = []
        human_encoder_layers.append(nn.Linear(actor_sh_dim, human_encoder_hidden_dims[0]))
        human_encoder_layers.append(activation_fn)
        for i in range(len(human_encoder_hidden_dims) - 1):
            human_encoder_layers.append(nn.Linear(human_encoder_hidden_dims[i], human_encoder_hidden_dims[i+1]))
            human_encoder_layers.append(activation_fn)
        human_encoder_layers.append(nn.Linear(human_encoder_hidden_dims[-1], fsqvae_latent_dim))
        self.human_encoder = nn.Sequential(*human_encoder_layers)

        # FSQ Quantizer: Latent -> Quantized Latent
        self.fsq = FSQ(levels=levels, dim=fsqvae_latent_dim, num_codebooks=num_codebooks)

        # Proprioceptive State Normalizer
        actor_sp_dim = num_actor_obs - actor_sg_dim - actor_sh_dim
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(actor_sp_dim,))

        # Action Decoder: Quantized Latent + Proprioceptive State -> Actions
        action_decoder_input_dim = fsqvae_latent_dim + (num_actor_obs - actor_sg_dim - actor_sh_dim)
        action_layers = []
        action_layers.append(nn.Linear(action_decoder_input_dim, action_decoder_hidden_dims[0]))
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
        if self.activate_signals == "robot":
            actions = self.forward_robot(x)
        elif self.activate_signals == "smplx":
            actions = self.forward_smplx(x)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}. Must be 'robot' or 'smplx'.")

        return actions
    
    def forward_robot(self, x):
        # Split input into sg and sp
        x_sg = x[:, :self.actor_sg_dim]
        x_sp = x[:, self.actor_sg_dim+self.actor_sh_dim:]
        
        # Encode sg to latent
        h_sg = self.robot_encoder(x_sg)
        z_sg, indices_sg = self.fsq(h_sg)
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sg, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_smplx(self, x):
        # Split input into sg and sp
        x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Encode sh to latent
        h_sh = self.human_encoder(x_sh)
        z_sh, indices_sh = self.fsq(h_sh)
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sh, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_robot_exporter(self, robot_command, proprioceptive_state):
        """Forward pass for robot command and proprioceptive state.
        
        Args:
            robot_command: Input tensor of shape (batch_size, actor_sg_dim)
            proprioceptive_state: Input tensor of shape (batch_size, num_actor_obs - actor_sg_dim - actor_sh_dim)
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Encode sg to latent
        h_sg = self.robot_encoder(robot_command)
        z_sg, indices_sg = self.fsq(h_sg)
        
        # Normalize proprioceptive state
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sg, proprioceptive_state_normalized], dim=-1))
        
        return actions
    
    def forward_smplx_exporter(self, smplx_human_state, proprioceptive_state):
        """Forward pass for smplx human state and proprioceptive state.
        
        Args:
            smplx_human_state: Input tensor of shape (batch_size, actor_sh_dim)
            proprioceptive_state: Input tensor of shape (batch_size, num_actor_obs - actor_sg_dim - actor_sh_dim)
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Encode sh to latent
        h_sh = self.human_encoder(smplx_human_state)
        z_sh, indices_sh = self.fsq(h_sh)
        
        # Normalize proprioceptive state
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sh, proprioceptive_state_normalized], dim=-1))
        
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
        x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        
        h_sg = self.robot_encoder(x_sg)
        h_sh = self.human_encoder(x_sh)
        
        z_sg, indices_sg = self.fsq(h_sg)
        z_sh, indices_sh = self.fsq(h_sh)

        return z_sg, z_sh
    
    def encode_robot(self, x):
        """Encode input robot goal state to quantized latent space.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
            
            indices: Quantization indices from FSQ
        """
        x_sg = x[:, :self.actor_sg_dim]
        
        h_sg = self.robot_encoder(x_sg)
        
        z_sg, indices_sg = self.fsq(h_sg)

        return z_sg, indices_sg
    
    def encode_smplx(self, x):
        """Encode input smplx human state to quantized latent space.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
            indices: Quantization indices from FSQ
        """
        x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        
        h_sh = self.human_encoder(x_sh)
        
        z_sh, indices_sh = self.fsq(h_sh)

        return z_sh, indices_sh

    def decode_action(self, z_q, x_sp):
        """Decode actions from quantized latent and proprioceptive state.
        
        Args:
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, num_actor_obs - actor_sg_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
            
        NOTE: 
            z_q both supports robot and human quantized latents.
        """
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        return self.action_decoder(torch.cat([z_q, x_sp_normalized], dim=-1))
    

    def decode_recover(self, z_q):
        """Decode reconstruction of goal state from quantized latent.
        
        Args:
            z_q: Quantized latent vector of shape (batch_size, fsqvae_latent_dim)
        
        Returns:
            recon: Reconstructed goal state of shape (batch_size, actor_sg_dim)
        
        NOTE: 
            z_q both suports robot and human quantized latents.    
        
        """
        return self.recover_decoder(z_q)
    
    ############################################
    # Freeze and Unfreeze modules
    ###########################################
    
    # def freeze_encoders_and_decoders(self):
    #     """Freeze all encoder and decoder modules, only train action_decoder.
        
    #     This method:
    #     1. Sets frozen modules to eval() mode to disable dropout/batch norm
    #     2. Disables gradient computation for frozen parameters
    #     3. Enables training for action_decoder only
    #     4. Keeps proprioceptive_normalizer in training mode to update statistics
        
    #     Call this once before training starts.
    #     """
    #     # Modules to freeze
    #     frozen_modules:list[nn.Module] = [
    #         self.robot_encoder,
    #         self.human_encoder,
    #         self.fsq,
    #         self.recover_decoder,
    #     ]
        
    #     # Set to eval mode and freeze gradients
    #     for module in frozen_modules:
    #         module.eval()
    #         for param in module.parameters():
    #             param.requires_grad = False
        
    #     # Ensure action_decoder is trainable
    #     self.action_decoder.train()
    #     for param in self.action_decoder.parameters():
    #         param.requires_grad = True
        
    #     # Keep proprioceptive_normalizer in training mode to update normalization statistics
    #     self.proprioceptive_normalizer.train()
            
    def train(self, mode: bool = True):
        """Override train() to ensure frozen modules stay in eval mode.
        
        The proprioceptive_normalizer is kept in training mode to continuously
        update its normalization statistics.
        """
        super().train(mode)
        # Ensure frozen modules remain in eval mode
        if mode:
            frozen_modules:list[nn.Module] = [
                self.robot_encoder,
                self.fsq,
                self.action_decoder,
                self.recover_decoder,
                self.proprioceptive_normalizer,
            ]
            for module in frozen_modules:
                module.eval()
            
    def freeze_for_finetune(self):
        # Freeze all modules for finetuning except self.human_encoder
        frozen_modules:list[nn.Module] = [
            self.robot_encoder,
            self.fsq,
            self.action_decoder,
            self.recover_decoder,
            self.proprioceptive_normalizer,
        ]
        for module in frozen_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        
        # Ensure human_encoder is trainable
        self.human_encoder.train()
        for param in self.human_encoder.parameters():
            param.requires_grad = True